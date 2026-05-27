"""
Vector Database Setup for Product Embeddings
Uses FAISS + Jina AI for efficient similarity search
"""
import json
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import pickle
import os

from app.schemas.schemas import Product
from app.services.jina_embedder import JinaEmbedder

@dataclass
class ProductDocument:
    """Structured product document for embedding"""
    ticker: str
    name: str
    description: str
    risk_score: int
    asset_class: str
    min_investment: float
    metadata: Dict

class VectorStore:
    """
    In-memory vector store using FAISS
    Supports Jina AI embeddings for production use
    """
    
    def __init__(self, embedding_dim: int = 1024):
        """
        Initialize vector store with FAISS
        """
        try:
            import faiss
            self.faiss = faiss
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.documents = []
            self.embedding_dim = embedding_dim
            print(f"✅ Vector store initialized (dimension: {embedding_dim})")
        except ImportError:
            raise ImportError(
                "FAISS not installed. Run: pip install faiss-cpu"
            )
    
    def add_documents(
        self, 
        documents: List[ProductDocument], 
        embeddings: np.ndarray
    ):
        """
        Add documents and their embeddings to the index
        """
        if len(documents) != len(embeddings):
            raise ValueError("Document count must match embedding count")
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        print(f"✅ Added {len(documents)} documents to vector store")
        print(f"   Total documents: {len(self.documents)}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Search for similar documents using pre-computed query embedding
        """
        if len(self.documents) == 0:
            print("⚠️  No documents in vector store")
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        similarities = -distances[0]
        
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(similarity)))
        
        return results
    
    def get_all_documents(self) -> List[ProductDocument]:
        """Get all documents in the store"""
        return self.documents
    
    def save(self, filepath: str):
        """
        Save vector store to disk
        """
        self.faiss.write_index(self.index, f"{filepath}.index")
        with open(f"{filepath}.docs", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"✅ Vector store saved to {filepath}.index and {filepath}.docs")
    
    def load(self, filepath: str):
        """
        Load vector store from disk
        """
        self.index = self.faiss.read_index(f"{filepath}.index")
        with open(f"{filepath}.docs", 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"✅ Vector store loaded from {filepath}")
        print(f"   Documents: {len(self.documents)}")
