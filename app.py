"""
Semantic Document Search API
Main Flask application for semantic search using BERT embeddings and FAISS
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List
import time

from models.bert_encoder import BERTEncoder
from models.faiss_index import FAISSIndex
from utils.preprocessing import preprocess_text
from utils.document_loader import DocumentLoader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize models (lazy loading)
bert_encoder = None
faiss_index = None
document_loader = None


def initialize_models():
    """Initialize ML models on first request"""
    global bert_encoder, faiss_index, document_loader
    
    if bert_encoder is None:
        logger.info("Initializing BERT encoder...")
        model_name = os.getenv('BERT_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        bert_encoder = BERTEncoder(model_name=model_name)
    
    if faiss_index is None:
        logger.info("Loading FAISS index...")
        index_path = os.getenv('FAISS_INDEX_PATH', 'data/faiss_index')
        faiss_index = FAISSIndex(index_path=index_path)
        faiss_index.load()
    
    if document_loader is None:
        logger.info("Initializing document loader...")
        document_loader = DocumentLoader()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'semantic-search-api',
        'version': '1.0.0'
    }), 200


@app.route('/api/search', methods=['POST'])
def search():
    """
    Search documents using semantic similarity
    
    Request Body:
        {
            "query": "search query text",
            "top_k": 10,
            "threshold": 0.7
        }
    """
    try:
        start_time = time.time()
        
        # Initialize models if needed
        initialize_models()
        
        # Parse request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data.get('query', '')
        top_k = data.get('top_k', 10)
        threshold = data.get('threshold', 0.0)
        
        # Validate inputs
        if not query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if top_k < 1 or top_k > 100:
            return jsonify({'error': 'top_k must be between 1 and 100'}), 400
        
        # Preprocess query
        processed_query = preprocess_text(query)
        
        # Generate query embedding
        query_embedding = bert_encoder.encode(processed_query)
        
        # Search in FAISS index
        distances, indices = faiss_index.search(query_embedding, top_k)
        
        # Retrieve document metadata
        results = []
        for idx, (distance, doc_idx) in enumerate(zip(distances[0], indices[0])):
            score = 1 - (distance / 2)  # Convert L2 distance to similarity score
            
            if score < threshold:
                continue
            
            doc_metadata = faiss_index.get_document(int(doc_idx))
            if doc_metadata:
                results.append({
                    'rank': idx + 1,
                    'document_id': doc_metadata.get('id'),
                    'title': doc_metadata.get('title'),
                    'content': doc_metadata.get('content', '')[:500],  # Snippet
                    'score': float(score),
                    'metadata': doc_metadata.get('metadata', {})
                })
        
        query_time = time.time() - start_time
        
        return jsonify({
            'results': results,
            'total_results': len(results),
            'query_time': round(query_time, 3),
            'query': query
        }), 200
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/index', methods=['POST'])
def index_document():
    """
    Index a new document
    
    Request Body:
        {
            "document_id": "doc_123",
            "title": "Document Title",
            "content": "Document content...",
            "metadata": {...}
        }
    """
    try:
        initialize_models()
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['document_id', 'title', 'content']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        document_id = data['document_id']
        title = data['title']
        content = data['content']
        metadata = data.get('metadata', {})
        
        # Preprocess content
        processed_content = preprocess_text(content)
        
        # Generate embedding
        embedding = bert_encoder.encode(processed_content)
        
        # Add to FAISS index
        doc_data = {
            'id': document_id,
            'title': title,
            'content': content,
            'metadata': metadata
        }
        
        faiss_index.add_document(embedding, doc_data)
        
        return jsonify({
            'message': 'Document indexed successfully',
            'document_id': document_id
        }), 201
    
    except Exception as e:
        logger.error(f"Indexing error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/similar/<document_id>', methods=['GET'])
def find_similar(document_id):
    """Find documents similar to a given document"""
    try:
        initialize_models()
        
        top_k = request.args.get('top_k', default=10, type=int)
        
        # Get document embedding
        doc_data = faiss_index.get_document_by_id(document_id)
        if not doc_data:
            return jsonify({'error': 'Document not found'}), 404
        
        content = doc_data.get('content', '')
        processed_content = preprocess_text(content)
        embedding = bert_encoder.encode(processed_content)
        
        # Search for similar documents
        distances, indices = faiss_index.search(embedding, top_k + 1)
        
        # Filter out the query document itself
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            doc = faiss_index.get_document(int(idx))
            if doc and doc.get('id') != document_id:
                score = 1 - (distance / 2)
                results.append({
                    'document_id': doc.get('id'),
                    'title': doc.get('title'),
                    'score': float(score)
                })
        
        return jsonify({
            'source_document_id': document_id,
            'similar_documents': results[:top_k]
        }), 200
    
    except Exception as e:
        logger.error(f"Similar search error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get index statistics"""
    try:
        initialize_models()
        
        stats = {
            'total_documents': faiss_index.get_total_documents(),
            'embedding_dimension': bert_encoder.get_dimension(),
            'model_name': bert_encoder.model_name,
            'index_type': faiss_index.index_type
        }
        
        return jsonify(stats), 200
    
    except Exception as e:
        logger.error(f"Stats error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Semantic Search API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
