"""
Compiled vector storage indexing script.
Generates FAISS database for similarity queries, saved to data/ directory.
"""
import json
import numpy as np
import os

from app.schemas.schemas import Product
from app.services.vector_store import VectorStore, ProductDocument
from app.services.jina_embedder import get_embedder

def build_vector_store():
    """
    Build vector store using real or mock Jina AI embeddings
    """
    print("\n" + "="*70)
    print("🏗️  BUILDING VECTOR STORE INDEX")
    print("="*70 + "\n")
    
    # Try getting the embedder
    embedder = get_embedder(use_real=True)
    
    # Load products from data/
    print("\n📦 Loading products from data/...")
    try:
        with open('data/product_master.json', 'r') as f:
            products_data = json.load(f)
            products = [Product(**p) for p in products_data]
        print(f"✅ Loaded {len(products)} products")
    except FileNotFoundError:
        print("❌ product_master.json not found inside data/!")
        print("   Generating mock data first...")
        from scripts.mock_data import save_to_json
        save_to_json()
        with open('data/product_master.json', 'r') as f:
            products_data = json.load(f)
            products = [Product(**p) for p in products_data]
    
    # Build descriptions
    print("\n📝 Building product descriptions...")
    descriptions = []
    product_docs = []
    
    for product in products:
        desc = f"{product.name}. "
        desc += f"Ticker: {product.ticker}. "
        desc += f"Asset class: {product.asset_class}. "
        desc += f"Risk score: {product.risk_score} out of 5. "
        desc += f"Minimum investment: ${product.min_investment:,.0f}."
        
        descriptions.append(desc)
        
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
    
    # Generate embeddings
    print("\n🔮 Generating embeddings...")
    embeddings = embedder.embed(descriptions)
    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    
    # Create vector store
    print("\n📊 Creating FAISS vector store...")
    vector_store = VectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add_documents(product_docs, embeddings)
    
    # Save vector store inside data/
    print("\n💾 Saving vector store...")
    os.makedirs('data', exist_ok=True)
    vector_store.save('data/product_vector_store')
    
    print("\n" + "="*70)
    print("✅ VECTOR STORE BUILD COMPLETE!")
    print("="*70)
    print(f"📊 Statistics:")
    print(f"   Products indexed: {len(products)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Files created:")
    print(f"      - data/product_vector_store.index")
    print(f"      - data/product_vector_store.docs")
    print("="*70)
    print()


if __name__ == "__main__":
    build_vector_store()
