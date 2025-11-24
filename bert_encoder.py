"""
BERT Encoder for Semantic Document Search
Converts text documents into dense vector embeddings using sentence-transformers
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BERTEncoder:
    """
    Encodes text documents using BERT-based sentence transformers.
    
    Features:
    - Batch processing for efficiency
    - GPU acceleration
    - Normalization for cosine similarity
    - Caching for repeated queries
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: int = 256,
        normalize_embeddings: bool = True
    ):
        """
        Initialize BERT encoder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
            max_seq_length: Maximum sequence length
            normalize_embeddings: Whether to L2 normalize embeddings
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing BERT encoder on {self.device}")
        
        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length
        
        # Embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar
            convert_to_numpy: Convert to numpy array
            
        Returns:
            Embeddings as numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Encode with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embeddings
    
    def encode_documents(
        self,
        documents: List[dict],
        text_field: str = "content",
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of document dictionaries.
        
        Args:
            documents: List of document dicts with text content
            text_field: Field name containing text
            show_progress: Show progress bar
            
        Returns:
            Document embeddings
        """
        texts = [doc[text_field] for doc in documents]
        
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self.encode(texts, show_progress=show_progress)
        
        return embeddings
    
    def encode_queries(
        self,
        queries: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Encode search queries.
        
        Args:
            queries: Single query or list of queries
            
        Returns:
            Query embeddings
        """
        return self.encode(queries, show_progress=False)
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def save_model(self, path: str):
        """Save the model to disk."""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None):
        """Load a saved model."""
        instance = cls.__new__(cls)
        instance.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        instance.model = SentenceTransformer(path, device=instance.device)
        instance.embedding_dim = instance.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded from {path}")
        return instance


class MultilingualBERTEncoder(BERTEncoder):
    """
    Multilingual BERT encoder supporting 50+ languages.
    """
    
    def __init__(self, **kwargs):
        kwargs['model_name'] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        super().__init__(**kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize encoder
    encoder = BERTEncoder()
    
    # Encode documents
    documents = [
        {"id": 1, "content": "Machine learning is a subset of artificial intelligence"},
        {"id": 2, "content": "Deep learning uses neural networks with multiple layers"},
        {"id": 3, "content": "Natural language processing enables computers to understand text"}
    ]
    
    doc_embeddings = encoder.encode_documents(documents)
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    
    # Encode query
    query = "What is deep learning?"
    query_embedding = encoder.encode_queries(query)
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Compute similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Get most similar document
    most_similar_idx = np.argmax(similarities)
    print(f"\nMost similar document: {documents[most_similar_idx]['content']}")
    print(f"Similarity score: {similarities[most_similar_idx]:.4f}")
