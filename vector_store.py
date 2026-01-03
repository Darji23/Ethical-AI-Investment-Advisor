"""
Vector Database Setup for Product Embeddings (Phase 2)
Uses FAISS + Jina AI for efficient similarity search
Works with 100 investors and 25 products from Phase 1
"""
import json
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import pickle
import os

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
        
        Args:
            embedding_dim: Dimension of embeddings (1024 for Jina AI)
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
        
        Args:
            documents: List of ProductDocument objects
            embeddings: Numpy array of embeddings (shape: [n_docs, embedding_dim])
        """
        if len(documents) != len(embeddings):
            raise ValueError("Document count must match embedding count")
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents
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
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of tuples: (document, similarity_score)
        """
        if len(self.documents) == 0:
            print("⚠️  No documents in vector store")
            return []
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Convert distances to similarity scores
        # Lower L2 distance = higher similarity
        similarities = -distances[0]
        
        # Get matching documents
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
        
        Args:
            filepath: Path to save (without extension)
        """
        # Save FAISS index
        self.faiss.write_index(self.index, f"{filepath}.index")
        
        # Save documents
        with open(f"{filepath}.docs", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"✅ Vector store saved to {filepath}.index and {filepath}.docs")
    
    def load(self, filepath: str):
        """
        Load vector store from disk
        
        Args:
            filepath: Path to load (without extension)
        """
        # Load FAISS index
        self.index = self.faiss.read_index(f"{filepath}.index")
        
        # Load documents
        with open(f"{filepath}.docs", 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"✅ Vector store loaded from {filepath}")
        print(f"   Documents: {len(self.documents)}")


# ============================================================================
# BUILD VECTOR STORE WITH JINA AI EMBEDDINGS
# ============================================================================

def build_vector_store_with_jina():
    """
    Build vector store using Jina AI embeddings
    Run this script to create the product vector store
    """
    print("\n" + "="*70)
    print("🏗️  BUILDING VECTOR STORE WITH JINA AI EMBEDDINGS")
    print("="*70 + "\n")
    
    # Import Jina embedder
    try:
        from jina_embedder import JinaEmbedder
        embedder = JinaEmbedder()
        print("✅ Using Jina AI embeddings")
    except Exception as e:
        print(f"❌ Failed to initialize Jina AI: {e}")
        print("   Make sure JINA_API_KEY is set!")
        print("   Run: export JINA_API_KEY='your-key-here'")
        return
    
    # Load products from Phase 1
    print("\n📦 Loading products from Phase 1...")
    try:
        from schemas import Product
        with open('product_master.json', 'r') as f:
            products_data = json.load(f)
            products = [Product(**p) for p in products_data]
        print(f"✅ Loaded {len(products)} products")
    except FileNotFoundError:
        print("❌ product_master.json not found!")
        print("   Run Phase 1 first: python mock_data_expanded.py")
        return
    
    # Build rich descriptions for embedding
    print("\n📝 Building product descriptions...")
    descriptions = []
    product_docs = []
    
    for product in products:
        # Create rich description
        desc = f"{product.name}. "
        desc += f"Ticker: {product.ticker}. "
        desc += f"Asset class: {product.asset_class}. "
        desc += f"Risk score: {product.risk_score} out of 5. "
        desc += f"Minimum investment: ${product.min_investment:,.0f}."
        
        descriptions.append(desc)
        
        # Create ProductDocument
        doc = ProductDocument(
            ticker=product.ticker,
            name=product.name,
            description=desc,
            risk_score=product.risk_score,
            asset_class=product.asset_class,
            min_investment=product.min_investment,
            metadata={
                "risk_score": product.risk_score,
                "asset_class": product.asset_class
            }
        )
        product_docs.append(doc)
    
    print(f"✅ Created {len(descriptions)} descriptions")
    
    # Generate embeddings with Jina AI
    print("\n🔮 Generating embeddings with Jina AI...")
    print("   (This may take 10-30 seconds...)")
    
    try:
        embeddings = embedder.embed(descriptions)
        print(f"✅ Generated embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dimension: {embeddings.shape[1]}")
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return
    
    # Create vector store
    print("\n📊 Creating FAISS vector store...")
    vector_store = VectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add_documents(product_docs, embeddings)
    
    # Save vector store
    print("\n💾 Saving vector store...")
    vector_store.save('product_vector_store')
    
    # Test semantic search
    print("\n🧪 Testing semantic search...")
    test_queries = [
        "I want safe conservative investments",
        "Give me high growth tech stocks",
        "I need stable income from bonds"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        # Embed query
        query_embedding = embedder.embed_query(query)
        
        # Search
        results = vector_store.search(query_embedding, top_k=3)
        
        print(f"   Top 3 results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"      {i}. {doc.ticker} - {doc.name[:40]:40} (score: {score:.3f})")
    
    print("\n" + "="*70)
    print("✅ VECTOR STORE BUILD COMPLETE!")
    print("="*70)
    print(f"📊 Statistics:")
    print(f"   Products indexed: {len(products)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Files created:")
    print(f"      - product_vector_store.index")
    print(f"      - product_vector_store.docs")
    print("="*70)
    print()


if __name__ == "__main__":
    """
    Run this script to build the vector store
    
    Prerequisites:
    1. Phase 1 completed (product_master.json exists)
    2. Jina API key set: export JINA_API_KEY='your-key'
    3. jina_embedder.py available
    
    Usage:
        python vector_store.py
    """
    build_vector_store_with_jina()
