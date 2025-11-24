"""
FAISS Index Manager for Fast Similarity Search
Handles vector indexing and retrieval using Facebook's FAISS library
"""

import faiss
import numpy as np
import pickle
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    Manages FAISS index for fast similarity search.
    
    Supports:
    - Flat index for exact search (small datasets)
    - IVF index for approximate search (large datasets)
    - GPU acceleration
    - Incremental updates
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "Flat",
        metric: str = "cosine",
        nlist: int = 100,
        use_gpu: bool = False
    ):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ('Flat', 'IVF', 'HNSW')
            metric: Distance metric ('cosine', 'l2', 'inner_product')
            nlist: Number of clusters for IVF index
            use_gpu: Whether to use GPU acceleration
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.use_gpu = use_gpu
        
        # Create index
        self.index = self._create_index()
        
        # Document metadata
        self.documents = []
        self.doc_ids = []
        
        logger.info(f"Initialized {index_type} FAISS index with dimension {embedding_dim}")
        
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        
        # Base index for cosine similarity
        if self.metric == "cosine":
            # Normalize embeddings for cosine similarity using inner product
            if self.index_type == "Flat":
                index = faiss.IndexFlatIP(self.embedding_dim)
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            elif self.index_type == "HNSW":
                index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                index.hnsw.efConstruction = 40
                index.hnsw.efSearch = 16
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
                
        elif self.metric == "l2":
            if self.index_type == "Flat":
                index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            
        return index
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: Optional[List[dict]] = None,
        doc_ids: Optional[List[Union[str, int]]] = None
    ):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Embedding vectors to add
            documents: Associated document metadata
            doc_ids: Document IDs
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Train index if needed (IVF requires training)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info(f"Training index on {len(embeddings)} vectors...")
            self.index.train(embeddings)
            
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        if documents:
            self.documents.extend(documents)
        if doc_ids:
            self.doc_ids.extend(doc_ids)
        else:
            # Generate IDs if not provided
            start_id = len(self.doc_ids)
            self.doc_ids.extend(range(start_id, start_id + len(embeddings)))
            
        logger.info(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        nprobe: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_embeddings: Query vectors
            top_k: Number of results to return
            nprobe: Number of clusters to search (for IVF)
            
        Returns:
            distances: Similarity scores
            indices: Indices of similar vectors
        """
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
            
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_embeddings)
        
        # Set nprobe for IVF index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
            
        # Search
        distances, indices = self.index.search(query_embeddings, top_k)
        
        return distances, indices
    
    def search_with_metadata(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        nprobe: int = 10
    ) -> List[List[dict]]:
        """
        Search and return results with document metadata.
        
        Args:
            query_embeddings: Query vectors
            top_k: Number of results per query
            nprobe: Number of clusters to search
            
        Returns:
            List of result lists, each containing dicts with score and document
        """
        distances, indices = self.search(query_embeddings, top_k, nprobe)
        
        results = []
        for query_idx in range(len(query_embeddings)):
            query_results = []
            for rank, (score, idx) in enumerate(zip(distances[query_idx], indices[query_idx])):
                if idx >= 0:  # Valid result
                    result = {
                        'rank': rank + 1,
                        'score': float(score),
                        'doc_id': self.doc_ids[idx] if idx < len(self.doc_ids) else idx,
                        'document': self.documents[idx] if idx < len(self.documents) else None
                    }
                    query_results.append(result)
            results.append(query_results)
            
        return results
    
    def remove(self, ids: List[int]):
        """
        Remove vectors by ID (only supported for some index types).
        
        Args:
            ids: List of vector IDs to remove
        """
        if hasattr(self.index, 'remove_ids'):
            id_selector = faiss.IDSelectorBatch(ids)
            self.index.remove_ids(id_selector)
            logger.info(f"Removed {len(ids)} vectors")
        else:
            logger.warning("Remove operation not supported for this index type")
    
    def save(self, path: str):
        """
        Save index and metadata to disk.
        
        Args:
            path: Directory path to save files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "index.faiss"
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata = {
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'nlist': self.nlist
        }
        
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        logger.info(f"Saved index to {path}")
    
    @classmethod
    def load(cls, path: str, use_gpu: bool = False):
        """
        Load index and metadata from disk.
        
        Args:
            path: Directory path containing saved files
            use_gpu: Whether to load index to GPU
            
        Returns:
            Loaded FAISSIndex instance
        """
        path = Path(path)
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            embedding_dim=metadata['embedding_dim'],
            index_type=metadata['index_type'],
            metric=metadata['metric'],
            nlist=metadata['nlist'],
            use_gpu=False  # Load on CPU first
        )
        
        # Load FAISS index
        index_path = path / "index.faiss"
        instance.index = faiss.read_index(str(index_path))
        
        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            instance.index = faiss.index_cpu_to_gpu(res, 0, instance.index)
            instance.use_gpu = True
        
        # Restore metadata
        instance.documents = metadata['documents']
        instance.doc_ids = metadata['doc_ids']
        
        logger.info(f"Loaded index from {path} with {instance.index.ntotal} vectors")
        return instance
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            'use_gpu': self.use_gpu
        }


# Example usage
if __name__ == "__main__":
    # Create sample embeddings
    embedding_dim = 384
    n_documents = 1000
    embeddings = np.random.randn(n_documents, embedding_dim).astype(np.float32)
    
    # Create documents
    documents = [
        {'id': i, 'title': f'Document {i}', 'content': f'Content {i}'}
        for i in range(n_documents)
    ]
    
    # Initialize and build index
    index = FAISSIndex(embedding_dim=embedding_dim, index_type="IVF")
    index.add(embeddings, documents=documents)
    
    # Search
    query = np.random.randn(1, embedding_dim).astype(np.float32)
    results = index.search_with_metadata(query, top_k=5)
    
    print("Search Results:")
    for result in results[0]:
        print(f"Rank {result['rank']}: Score={result['score']:.4f}, Doc={result['document']['title']}")
    
    # Save and load
    index.save("test_index")
    loaded_index = FAISSIndex.load("test_index")
    print(f"\nLoaded index stats: {loaded_index.get_stats()}")
