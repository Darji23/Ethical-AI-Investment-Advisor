"""
Jina AI Embedder - Real Embeddings for Production
Uses Jina AI's embedding API for high-quality semantic search
"""
import requests
import os
from typing import List
import numpy as np
from app.core.config import JINA_API_KEY

class JinaEmbedder:
    """
    Real embeddings using Jina AI API
    """
    
    def __init__(self, api_key: str = None, model: str = "jina-embeddings-v3"):
        """
        Initialize Jina embedder
        """
        self.api_key = api_key or JINA_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Jina API key required! Get one at https://jina.ai/ and set:\n"
                "  export JINA_API_KEY='your-key-here'\n"
                "  or pass api_key parameter"
            )
        
        self.model = model
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.embedding_dim = 1024
        
        print(f"✅ Jina AI Embedder initialized")
        print(f"   Model: {self.model}")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        """
        if not texts:
            return np.array([])
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "input": texts,
                "encoding_format": "float"
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Jina API error: {response.status_code} - {response.text}")
            
            data = response.json()
            embeddings = [item['embedding'] for item in data['data']]
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Jina embedding error: {e}")
            raise
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query
        """
        embeddings = self.embed([text])
        return embeddings[0]


class MockEmbedder:
    """
    Fallback mock embedder for testing without API key
    """
    
    def __init__(self, dim: int = 1024):
        """
        Initialize mock embedder
        """
        self.embedding_dim = dim
        print(f"⚠️  Using MockEmbedder (for testing only)")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate random embeddings (for testing)"""
        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    def embed_query(self, text: str) -> np.ndarray:
        """Generate random embedding for query (for testing)"""
        return np.random.randn(self.embedding_dim).astype(np.float32)


def get_embedder(use_real: bool = True) -> JinaEmbedder | MockEmbedder:
    """
    Get embedder instance (real or mock)
    """
    if use_real:
        try:
            return JinaEmbedder()
        except ValueError as e:
            print(f"⚠️  {e}")
            print(f"⚠️  Falling back to MockEmbedder")
            return MockEmbedder(dim=1024)
    else:
        return MockEmbedder(dim=1024)
