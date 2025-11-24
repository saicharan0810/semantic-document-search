# 🔍 Semantic Document Search Engine

A production-grade semantic search system powered by BERT embeddings and FAISS vector database, enabling intelligent document retrieval with natural language understanding.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![AWS](https://img.shields.io/badge/AWS-Lambda%20%7C%20S3-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Semantic Understanding**: BERT-based embeddings capture contextual meaning beyond keyword matching
- **Lightning-Fast Search**: FAISS vector database enables sub-second queries across 100K+ documents
- **89% Relevance Score**: Superior accuracy compared to traditional keyword-based search
- **RESTful API**: Flask-based backend with comprehensive API endpoints
- **Modern Frontend**: React-based UI with real-time search capabilities
- **Cloud-Native**: Serverless deployment on AWS Lambda with S3 storage
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   React UI  │────▶ │  Flask API   │────▶ │    BERT     │
└─────────────┘      └──────────────┘      │  Embeddings │
                            │               └─────────────┘
                            ▼                      │
                     ┌──────────────┐             ▼
                     │     S3       │      ┌─────────────┐
                     │   Storage    │      │    FAISS    │
                     └──────────────┘      │   Vector DB │
                                           └─────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 14+
- AWS Account (for deployment)
- 8GB+ RAM (for BERT model)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/semantic-document-search.git
cd semantic-document-search
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained BERT model
python scripts/download_model.py

# Start Flask server
python app.py
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm start
```

### 4. Access the Application

Open your browser and navigate to `http://localhost:3000`

## 📁 Project Structure

```
semantic-document-search/
├── backend/
│   ├── app.py                 # Flask application entry point
│   ├── models/
│   │   ├── bert_encoder.py    # BERT embedding model
│   │   └── faiss_index.py     # FAISS vector index
│   ├── api/
│   │   ├── routes.py          # API endpoints
│   │   └── validators.py      # Input validation
│   ├── utils/
│   │   ├── preprocessing.py   # Text preprocessing
│   │   └── document_loader.py # Document parsing
│   └── config.py              # Configuration settings
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── SearchBar.jsx
│   │   │   ├── ResultsCard.jsx
│   │   │   └── DocumentViewer.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   └── App.jsx
│   └── package.json
├── tests/
│   ├── test_embeddings.py
│   ├── test_search.py
│   └── test_api.py
├── scripts/
│   ├── index_documents.py     # Build FAISS index
│   └── benchmark.py           # Performance testing
├── deployment/
│   ├── lambda_function.py     # AWS Lambda handler
│   ├── serverless.yml         # Serverless config
│   └── Dockerfile
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions workflow
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔧 Configuration

Create a `.env` file in the root directory:

```env
# Flask Configuration
FLASK_ENV=development
FLASK_APP=app.py
SECRET_KEY=your-secret-key

# Model Configuration
BERT_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
MAX_SEQUENCE_LENGTH=512

# FAISS Configuration
FAISS_INDEX_TYPE=IndexFlatL2
NLIST=100

# AWS Configuration (for deployment)
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
LAMBDA_FUNCTION_NAME=semantic-search-api
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=backend tests/

# Run specific test suite
pytest tests/test_embeddings.py -v
```

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Index Size | 100,000 documents |
| Query Latency | < 100ms (p95) |
| Relevance Score | 89% accuracy |
| Throughput | 500 queries/second |
| Memory Usage | 4GB (index loaded) |

## 🔍 API Documentation

### Search Documents

```bash
POST /api/search
Content-Type: application/json

{
  "query": "machine learning optimization techniques",
  "top_k": 10,
  "threshold": 0.75
}
```

**Response:**

```json
{
  "results": [
    {
      "document_id": "doc_12345",
      "title": "Advanced ML Optimization",
      "content": "...",
      "score": 0.92,
      "metadata": {
        "author": "John Doe",
        "date": "2024-01-15"
      }
    }
  ],
  "query_time": 0.087,
  "total_results": 47
}
```

### Index New Document

```bash
POST /api/index
Content-Type: application/json

{
  "document_id": "doc_12346",
  "title": "Document Title",
  "content": "Document content...",
  "metadata": {
    "author": "Jane Smith",
    "category": "research"
  }
}
```

## 🚢 Deployment

### AWS Lambda Deployment

```bash
# Install Serverless Framework
npm install -g serverless

# Deploy to AWS
cd deployment
serverless deploy --stage production
```

### Docker Deployment

```bash
# Build Docker image
docker build -t semantic-search:latest .

# Run container
docker run -p 5000:5000 semantic-search:latest
```

## 🛠️ Technical Stack

**Backend:**
- Python 3.8+
- Flask 2.0
- Transformers (Hugging Face)
- FAISS (Facebook AI)
- sentence-transformers
- NumPy, Pandas

**Frontend:**
- React 18
- Axios
- Tailwind CSS
- React Query

**Infrastructure:**
- AWS Lambda
- AWS S3
- AWS API Gateway
- Docker
- GitHub Actions

## 📈 Roadmap

- [ ] Multi-language support
- [ ] Advanced filtering and faceted search
- [ ] Document similarity clustering
- [ ] Real-time index updates
- [ ] GraphQL API
- [ ] Elasticsearch integration
- [ ] Query auto-completion
- [ ] Analytics dashboard

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Sai Charan Kolluru** - [LinkedIn](https://linkedin.com/in/kscharan1608) | [Email](mailto:kscharan1608@gmail.com)

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Facebook AI Research for FAISS
- sentence-transformers community
- University of Maryland Baltimore County

## 📧 Contact

For questions or feedback, reach out at kscharan1608@gmail.com

---

⭐ If you find this project useful, please consider giving it a star!
