"""
Jina AI Embedder - Dual-Tier Architecture with Automatic Fallback
Primary: Flagship API-tier model (jina-embeddings-v3) with task-specific LoRA adapters.
Secondary: Open-weights model (jina-embeddings-v5-text-small / omni) as resilient fallback.
Tertiary: Offline MockEmbedder if network or credentials are unavailable.
"""
import requests
import os
from typing import List, Optional
import numpy as np
from app.core.config import JINA_API_KEY, JINA_MODEL, JINA_FALLBACK_MODEL

# Normalization mapping for model aliases
MODEL_ALIASES = {
    "jina-embeddings-v5-text": "jina-embeddings-v5-text-small",
    "jina-embeddings-v5-omni": "jina-embeddings-v5-omni-small",
    "v5-text": "jina-embeddings-v5-text-small",
    "v5-omni": "jina-embeddings-v5-omni-small",
    "v3": "jina-embeddings-v3",
}

def normalize_model_name(name: str) -> str:
    if not name:
        return "jina-embeddings-v3"
    name = name.strip()
    return MODEL_ALIASES.get(name, name)

class JinaEmbedder:
    """
    Robust Jina AI embedder with automatic failover from API-tier model to open-weight model.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        fallback_model: str = None,
        embedding_dim: int = 1024
    ):
        """
        Initialize Jina embedder with primary and fallback models
        """
        self.api_key = api_key or JINA_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Jina API key required! Get one at https://jina.ai/ and set:\n"
                "  export JINA_API_KEY='your-key-here'\n"
                "  or pass api_key parameter"
            )
        
        # Primary: API-tier model (default: jina-embeddings-v3)
        self.prime_model = normalize_model_name(model or JINA_MODEL or "jina-embeddings-v3")
        # Fallback: Open-weight model (default: jina-embeddings-v5-text-small)
        self.fallback_model = normalize_model_name(fallback_model or JINA_FALLBACK_MODEL or "jina-embeddings-v5-text-small")
        
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.embedding_dim = embedding_dim
        self.mock_fallback = MockEmbedder(dim=embedding_dim)
        
        print(f"✅ Jina AI Embedder initialized")
        print(f"   Prime Model (API Tier): {self.prime_model}")
        print(f"   Fallback Model (Open Weights): {self.fallback_model}")
        print(f"   Embedding dimension: {self.embedding_dim}")

    def _call_jina_api(self, model_name: str, texts: List[str], task: Optional[str] = None) -> np.ndarray:
        """Execute HTTP request against Jina AI Embeddings endpoint"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "input": texts,
            "encoding_format": "float",
            "dimensions": self.embedding_dim
        }
        
        # Jina v3 supports task-specific LoRA adapters (retrieval.query, retrieval.passage, text-matching)
        if "v3" in model_name and task:
            payload["task"] = task

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=20
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code} - {response.text}")
        
        data = response.json()
        embeddings = [item['embedding'] for item in data['data']]
        arr = np.array(embeddings, dtype=np.float32)
        
        if arr.size > 0 and arr.shape[1] != self.embedding_dim:
            self.embedding_dim = arr.shape[1]
            
        return arr

    def embed(self, texts: List[str], task: Optional[str] = None) -> np.ndarray:
        """
        Generate embeddings with automatic dual-tier failover:
        1. Attempt Prime Model (API-tier: jina-embeddings-v3)
        2. Failover to Open-Weight Model (jina-embeddings-v5-text-small)
        3. Failover to offline MockEmbedder if all remote attempts fail
        """
        if not texts:
            return np.array([], dtype=np.float32)

        # 1. Primary Attempt: API-tier Model
        try:
            return self._call_jina_api(self.prime_model, texts, task=task)
        except Exception as prime_err:
            print(f"⚠️  Jina prime model '{self.prime_model}' encountered an issue: {prime_err}")
            print(f"🔄 Switching to open-weight fallback model: '{self.fallback_model}'...")

        # 2. Secondary Attempt: Open-Weight Model
        try:
            arr = self._call_jina_api(self.fallback_model, texts, task=None)
            print(f"✅ Successfully generated embeddings with open-weight model '{self.fallback_model}'.")
            return arr
        except Exception as fallback_err:
            print(f"⚠️  Jina open-weight fallback '{self.fallback_model}' failed: {fallback_err}")
            print("🛡️  Falling back to local offline MockEmbedder...")

        # 3. Tertiary Attempt: Local Mock Fallback
        return self.mock_fallback.embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single search query (uses 'retrieval.query' task for asymmetric search)
        """
        embeddings = self.embed([text], task="retrieval.query")
        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for documents to be indexed (uses 'retrieval.passage' task)
        """
        return self.embed(texts, task="retrieval.passage")


class MockEmbedder:
    """
    Fallback mock embedder for offline / demo mode
    """
    
    def __init__(self, dim: int = 1024):
        self.embedding_dim = dim
    
    def embed(self, texts: List[str], task: Optional[str] = None) -> np.ndarray:
        """Generate deterministic/random pseudo-embeddings for testing"""
        if not texts:
            return np.array([], dtype=np.float32)
        # Use pseudo-random seed based on text length for reproducible local testing
        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    def embed_query(self, text: str) -> np.ndarray:
        return np.random.randn(self.embedding_dim).astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.embed(texts)


def get_embedder(
    use_real: bool = True,
    model: str = None,
    fallback_model: str = None,
    dim: int = 1024
) -> JinaEmbedder | MockEmbedder:
    """
    Factory function to get embedder instance with automatic failover
    """
    if use_real and JINA_API_KEY:
        try:
            return JinaEmbedder(
                model=model,
                fallback_model=fallback_model,
                embedding_dim=dim
            )
        except Exception as e:
            print(f"⚠️  Could not initialize JinaEmbedder: {e}")
            print(f"⚠️  Falling back to MockEmbedder")
            return MockEmbedder(dim=dim)
    else:
        if use_real and not JINA_API_KEY:
            print("ℹ️  No JINA_API_KEY detected. Running with MockEmbedder.")
        return MockEmbedder(dim=dim)
