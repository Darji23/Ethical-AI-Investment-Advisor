"""
Jina AI Embedder - Real Embeddings for Production
Uses Jina AI's embedding API for high-quality semantic search

Jina AI provides state-of-the-art embeddings optimized for search and RAG
"""
import requests
import os
from typing import List
import numpy as np

class JinaEmbedder:
    """
    Real embeddings using Jina AI API
    
    Jina AI offers:
    - 8192 dimension embeddings
    - Optimized for search and RAG
    - Multilingual support
    - Fast inference
    """
    
    def __init__(self, api_key: str = None, model: str = "jina-embeddings-v3"):
        """
        Initialize Jina embedder
        
        Args:
            api_key: Jina AI API key (get from https://jina.ai/)
            model: Model to use (default: jina-embeddings-v3)
        """
        self.api_key = api_key or os.getenv('JINA_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Jina API key required! Get one at https://jina.ai/ and set:\n"
                "  export JINA_API_KEY='your-key-here'\n"
                "  or pass api_key parameter"
            )
        
        self.model = model
        self.api_url = "https://api.jina.ai/v1/embeddings"
        
        # Jina embeddings are 1024 dimensions by default
        # (can be 768, 1024, or 8192 depending on model)
        self.embedding_dim = 1024
        
        print(f"✅ Jina AI Embedder initialized")
        print(f"   Model: {self.model}")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        try:
            # Call Jina API
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
            
            # Extract embeddings from response
            data = response.json()
            embeddings = [item['embedding'] for item in data['data']]
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Jina embedding error: {e}")
            raise
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            text: Query text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        embeddings = self.embed([text])
        return embeddings[0]


class MockEmbedder:
    """
    Fallback mock embedder for testing without API key
    (Keep this for backward compatibility)
    """
    
    def __init__(self, dim: int = 1024):
        """
        Initialize mock embedder
        
        Args:
            dim: Embedding dimension (default: 1024 to match Jina)
        """
        self.embedding_dim = dim
        print(f"⚠️  Using MockEmbedder (for testing only)")
        print(f"   For production, use JinaEmbedder with real API key")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate random embeddings (for testing)"""
        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    def embed_query(self, text: str) -> np.ndarray:
        """Generate random embedding for query (for testing)"""
        return np.random.randn(self.embedding_dim).astype(np.float32)


def get_embedder(use_real: bool = True) -> JinaEmbedder | MockEmbedder:
    """
    Get embedder instance (real or mock)
    
    Args:
        use_real: If True, try to use Jina AI (requires API key)
                 If False, use MockEmbedder
    
    Returns:
        JinaEmbedder or MockEmbedder instance
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


# =========================================================================
# TESTING SCRIPT
# =========================================================================

if __name__ == "__main__":
    """
    Test Jina embedder
    
    Usage:
        export JINA_API_KEY='your-key-here'
        python jina_embedder.py
    """
    print("\n🧪 TESTING JINA EMBEDDER\n")
    
    # Test texts
    texts = [
        "US Treasury Bond Fund - Conservative fixed income",
        "Total Stock Market Index - Broad market exposure",
        "Bitcoin ETF - High risk cryptocurrency exposure"
    ]
    
    # Try real embedder
    try:
        print("📊 Testing Jina AI Embeddings:")
        embedder = JinaEmbedder()
        
        embeddings = embedder.embed(texts)
        
        print(f"✅ Generated embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dimension: {embeddings.shape[1]}")
        print(f"   Sample values: {embeddings[0][:5]}")
        
        # Test query embedding
        query = "I want safe conservative investments"
        query_emb = embedder.embed_query(query)
        print(f"\n✅ Query embedding generated")
        print(f"   Shape: {query_emb.shape}")
        
        # Calculate similarity
        from numpy.linalg import norm
        
        print(f"\n📊 Similarity to query '{query}':")
        for i, text in enumerate(texts):
            # Cosine similarity
            similarity = np.dot(query_emb, embeddings[i]) / (norm(query_emb) * norm(embeddings[i]))
            print(f"   {text[:50]:50} | Similarity: {similarity:.4f}")
        
        print("\n✅ JINA EMBEDDER TESTS COMPLETE!\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTo use Jina embeddings:")
        print("1. Get API key from https://jina.ai/")
        print("2. Set environment variable:")
        print("   export JINA_API_KEY='your-key-here'")
        print("3. Run this script again")
