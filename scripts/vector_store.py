"""
Compiled vector storage indexing script.
Generates FAISS database for similarity queries, saved to data/ directory.
Loads products from MongoDB (sole data source).
"""
import json
import numpy as np
import os

from app.schemas.schemas import Product
from app.services.vector_store import VectorStore, ProductDocument
from app.services.jina_embedder import get_embedder

def build_vector_store():
    """
    Build vector store using real or mock Jina AI embeddings.
    Products are loaded from MongoDB.
    """
    print("\n" + "="*70)
    print("🏗️  BUILDING VECTOR STORE INDEX")
    print("="*70 + "\n")
    
    # Try getting the embedder
    embedder = get_embedder(use_real=True)
    
    # Load products from MongoDB
    print("\n📦 Loading products from MongoDB...")
    products = []
    try:
        from app.core.database import DatabaseManager
        db = DatabaseManager()
        if db.is_available():
            products = db.get_all_products()
            print(f"✅ Loaded {len(products)} products from MongoDB")
            db.close()
        else:
            print("❌ MongoDB not available!")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
    
    if not products:
        print("❌ No products found. Run 'python3 -m scripts.ingest_data --products' first.")
        return
    
    # Build descriptions
    print("\n📝 Building product descriptions...")
    descriptions = []
    product_docs = []
    
    for product in products:
        # Prefer the real Yahoo description if available; fall back to structured summary
        desc = product.description or ""
        if len(desc) < 20:
            lev_str = f" with {product.leverage_factor}x leverage" if (product.is_leveraged or product.is_inverse) else ""
            desc = (
                f"{product.name} ({product.ticker}) is an {product.asset_class} ETF{lev_str}, "
                f"risk {product.risk_score}/5, annual volatility {product.volatility_pct:.1f}%."
            )
        else:
            lev_str = f", {product.leverage_factor:+.0f}x leverage" if (product.is_leveraged or product.is_inverse) else ""
            desc = (
                f"{product.name} ({product.ticker}) — "
                f"{product.asset_class}{lev_str}, "
                f"risk {product.risk_score}/5. {desc}"
            )
        
        descriptions.append(desc)
        
        doc = ProductDocument(
            ticker=product.ticker,
            name=product.name,
            description=desc,
            risk_score=product.risk_score,
            asset_class=product.asset_class,
            min_investment=product.share_price,
            share_price=product.share_price,
            metadata={
                "risk_score":        product.risk_score,
                "asset_class":       product.asset_class,
                "volatility_pct":    product.volatility_pct,
                "is_leveraged":      product.is_leveraged,
                "is_inverse":        product.is_inverse,
                "leverage_factor":   product.leverage_factor,
                "srri":              product.srri,
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
    
    # Save vector store inside data/ with model name provenance
    model_name = getattr(embedder, "prime_model", "jina-embeddings-v3")
    print("\n💾 Saving vector store...")
    os.makedirs('data', exist_ok=True)
    vector_store.save('data/product_vector_store', model_name=model_name)
    
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
