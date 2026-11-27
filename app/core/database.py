"""
Database Layer - MongoDB Integration
Provides functions to save and retrieve data from MongoDB collections (investor_profiles and products).
MongoDB is the sole data source — no JSON fallbacks.
"""
from pymongo import MongoClient, ReplaceOne
from typing import List, Optional, Dict, Any
import os

from app.schemas.schemas import InvestorProfile, Product
from app.core.config import MONGODB_URL, MONGODB_DB_NAME

class DatabaseManager:
    """Manages MongoDB database connections and collections for Investors and Products"""
    
    def __init__(self, mongo_url: str = None, db_name: str = None):
        """
        Initialize MongoDB database connection
        """
        if mongo_url is None:
            mongo_url = MONGODB_URL
        if db_name is None:
            db_name = MONGODB_DB_NAME
            
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False
        
        try:
            self.client = MongoClient(mongo_url, serverSelectionTimeoutMS=2000)
            self.db = self.client[db_name]
            
            # Collections ("tables" in MongoDB cluster)
            self.investors_col = self.db["investor_profiles"]
            self.products_col = self.db["products"]
            
            # Quick ping to verify connection
            self.client.admin.command('ping')
            self.connected = True
            
            # Ensure unique indexes
            self._create_indexes()
            print(f"✅ MongoDB Database connected (Database: {db_name})")
        except Exception as e:
            print(f"⚠️  MongoDB Database not available ({e}). MongoDB is required — please ensure it is running.")
            self.client = None
            self.db = None
            self.connected = False

    def _create_indexes(self):
        """Create indexes for fast lookups and consistency"""
        if not self.connected:
            return
        try:
            self.investors_col.create_index("investor_id", unique=True)
            self.products_col.create_index("ticker", unique=True)
        except Exception as e:
            print(f"⚠️  Could not create MongoDB database indexes: {e}")

    @property
    def pool(self):
        """Backward compatibility for code checking db_manager.pool"""
        return self.client if self.connected else None

    def is_available(self) -> bool:
        """Check if MongoDB database is connected"""
        return self.connected and self.client is not None

    def close_all(self):
        """Close MongoDB connection"""
        self.close()

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            try:
                self.client.close()
                print("✅ MongoDB database connection closed")
            except Exception as e:
                print(f"⚠️  Error closing MongoDB connection: {e}")
            self.client = None
            self.connected = False

    # =========================================================================
    # INVESTOR PROFILE OPERATIONS
    # =========================================================================

    def save_investor_profile(self, profile: InvestorProfile) -> bool:
        """
        Save or update an investor profile in MongoDB
        """
        if not self.is_available():
            return False
        try:
            data = profile.dict() if hasattr(profile, 'dict') else profile.model_dump()
            self.investors_col.replace_one(
                {"investor_id": profile.investor_id},
                data,
                upsert=True
            )
            return True
        except Exception as e:
            print(f"❌ Failed to save investor profile {profile.investor_id}: {e}")
            return False

    def get_investor_profile(self, investor_id: str) -> Optional[InvestorProfile]:
        """
        Retrieve an investor profile from MongoDB
        """
        if not self.is_available():
            return None
        try:
            doc = self.investors_col.find_one({"investor_id": investor_id}, {"_id": 0})
            if doc:
                clean = {k: v for k, v in doc.items() if not k.startswith("_")}
                return InvestorProfile(**clean)
            return None
        except Exception as e:
            print(f"❌ Failed to fetch investor profile {investor_id}: {e}")
            return None

    def count_investors(self) -> int:
        """Total number of investor profiles (uses collection metadata, O(1))."""
        if not self.is_available():
            return 0
        try:
            return self.investors_col.estimated_document_count()
        except Exception as e:
            print(f"❌ Failed to count investors: {e}")
            return 0

    def get_all_investors(self, limit: int = None) -> List[InvestorProfile]:
        """
        Retrieve all investor profiles from MongoDB
        """
        if not self.is_available():
            return []
        try:
            cursor = self.investors_col.find({}, {"_id": 0})
            if limit:
                cursor = cursor.limit(limit)
            investors = []
            for doc in cursor:
                clean = {k: v for k, v in doc.items() if not k.startswith("_")}
                investors.append(InvestorProfile(**clean))
            return investors
        except Exception as e:
            print(f"❌ Failed to fetch all investors from MongoDB: {e}")
            return []

    # =========================================================================
    # PRODUCT OPERATIONS
    # =========================================================================

    def save_product(self, product: Product) -> bool:
        """
        Save or update a product in MongoDB
        """
        if not self.is_available():
            return False
        try:
            data = product.dict() if hasattr(product, 'dict') else product.model_dump()
            self.products_col.replace_one(
                {"ticker": product.ticker},
                data,
                upsert=True
            )
            return True
        except Exception as e:
            print(f"❌ Failed to save product {product.ticker}: {e}")
            return False

    def get_product(self, ticker: str) -> Optional[Product]:
        """
        Retrieve a product from MongoDB
        """
        if not self.is_available():
            return None
        try:
            doc = self.products_col.find_one({"ticker": ticker}, {"_id": 0})
            if doc:
                clean = {k: v for k, v in doc.items() if not k.startswith("_")}
                return Product(**clean)
            return None
        except Exception as e:
            print(f"❌ Failed to fetch product {ticker}: {e}")
            return None

    def get_all_products(self, active_only: bool = True) -> List[Product]:
        """
        Retrieve all products from MongoDB
        """
        if not self.is_available():
            return []
        try:
            query = {"active": True} if active_only else {}
            cursor = self.products_col.find(query, {"_id": 0})
            products = []
            for doc in cursor:
                clean = {k: v for k, v in doc.items() if not k.startswith("_")}
                products.append(Product(**clean))
            return products
        except Exception as e:
            print(f"❌ Failed to fetch products from MongoDB: {e}")
            return []

    # =========================================================================
    # DATA SEEDING
    # =========================================================================

    def seed_mock_data(self):
        """
        Seed MongoDB collections with mock data directly in MongoDB.
        For production ETF ingestion, use 'python3 -m scripts.ingest_data'.
        """
        if not self.is_available():
            return
        
        try:
            from scripts.mock_data import seed_mongodb
            seed_mongodb()
        except Exception as e:
            print(f"⚠️  Error seeding mock data to MongoDB: {e}")
