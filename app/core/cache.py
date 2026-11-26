"""
MongoDB Cache Layer
Provides high-performance caching for investors, products, and recommendation queries
using a dedicated 'cache' collection in the same MongoDB cluster, with automatic TTL expiration.
Handles offline MongoDB gracefully with an in-memory dictionary cache fallback.
"""
from pymongo import MongoClient, ReplaceOne
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import hashlib
import json
import os

from app.schemas.schemas import InvestorProfile, Product
from app.core.config import MONGODB_URL, MONGODB_DB_NAME

class MongoCache:
    """MongoDB-backed cache manager with automatic TTL index and in-memory fallback"""
    
    def __init__(self, mongo_url: str = None, db_name: str = None):
        """
        Initialize MongoDB cache collection
        """
        if mongo_url is None:
            mongo_url = MONGODB_URL
        if db_name is None:
            db_name = MONGODB_DB_NAME
            
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.client = None
        self.db = None
        self.cache_col = None
        self.connected = False
        
        # In-memory metrics & fallback cache
        self.hits = 0
        self.misses = 0
        self.local_fallback = {}
        
        try:
            self.client = MongoClient(mongo_url, serverSelectionTimeoutMS=2000)
            self.db = self.client[db_name]
            self.cache_col = self.db["cache"]
            
            # Quick ping to verify
            self.client.admin.command('ping')
            self.connected = True
            
            # Ensure TTL index on expires_at for automatic MongoDB document expiration
            self.cache_col.create_index("expires_at", expireAfterSeconds=0)
            self.cache_col.create_index("key", unique=True)
            print(f"✅ MongoDB Cache connected (Table/Collection: {db_name}.cache)")
        except Exception as e:
            print(f"⚠️  MongoDB Cache not available ({e}). Using in-memory cache fallback.")
            self.client = None
            self.connected = False

    def is_available(self) -> bool:
        """Check if MongoDB cache is available"""
        return self.connected and self.client is not None

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            try:
                self.client.close()
                print("✅ MongoDB cache connection closed")
            except Exception as e:
                print(f"⚠️  Error closing MongoDB cache: {e}")
            self.client = None
            self.connected = False

    # =========================================================================
    # GENERIC GET / SET
    # =========================================================================

    def _set(self, key: str, data: Any, type_name: str, ttl_seconds: int = 3600) -> bool:
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=ttl_seconds)
        doc = {
            "key": key,
            "type": type_name,
            "data": data,
            "created_at": now,
            "expires_at": expires_at
        }
        
        if self.is_available():
            try:
                self.cache_col.replace_one({"key": key}, doc, upsert=True)
                return True
            except Exception as e:
                print(f"⚠️  MongoDB cache set failed for {key}: {e}")
                
        # Store in local fallback
        self.local_fallback[key] = (expires_at, data)
        return True

    def _get(self, key: str) -> Optional[Any]:
        now = datetime.utcnow()
        if self.is_available():
            try:
                doc = self.cache_col.find_one({
                    "key": key,
                    "expires_at": {"$gt": now}
                })
                if doc:
                    self.hits += 1
                    return doc.get("data")
            except Exception as e:
                print(f"⚠️  MongoDB cache get failed for {key}: {e}")
                
        # Check local fallback
        if key in self.local_fallback:
            exp, data = self.local_fallback[key]
            if exp > now:
                self.hits += 1
                return data
            else:
                del self.local_fallback[key]
                
        self.misses += 1
        return None

    # =========================================================================
    # INVESTOR PROFILE CACHING
    # =========================================================================

    def cache_investor(self, investor: InvestorProfile, ttl_seconds: int = 3600) -> bool:
        key = f"investor:{investor.investor_id}"
        data = investor.dict() if hasattr(investor, 'dict') else investor.model_dump()
        return self._set(key, data, "investor", ttl_seconds)

    def get_investor(self, investor_id: str) -> Optional[InvestorProfile]:
        key = f"investor:{investor_id}"
        data = self._get(key)
        if data:
            return InvestorProfile(**data)
        return None

    def cache_all_investors(self, investors: List[InvestorProfile], ttl_seconds: int = 3600) -> int:
        count = 0
        for inv in investors:
            if self.cache_investor(inv, ttl_seconds):
                count += 1
        return count

    # =========================================================================
    # PRODUCT CACHING
    # =========================================================================

    def cache_product(self, product: Product, ttl_seconds: int = 7200) -> bool:
        key = f"product:{product.ticker}"
        data = product.dict() if hasattr(product, 'dict') else product.model_dump()
        return self._set(key, data, "product", ttl_seconds)

    def get_product(self, ticker: str) -> Optional[Product]:
        key = f"product:{ticker}"
        data = self._get(key)
        if data:
            return Product(**data)
        return None

    def cache_all_products(self, products: List[Product], ttl_seconds: int = 7200) -> int:
        count = 0
        for prod in products:
            if self.cache_product(prod, ttl_seconds):
                count += 1
        return count

    # =========================================================================
    # RECOMMENDATION CACHING
    # =========================================================================

    def _hash_query(self, investor_id: str, query: str) -> str:
        content = f"{investor_id}:{query.strip().lower()}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def cache_recommendation(self, investor_id: str, query: str, recommendation: Dict, ttl_seconds: int = 1800) -> bool:
        query_hash = self._hash_query(investor_id, query)
        key = f"rec:{query_hash}"
        return self._set(key, recommendation, "recommendation", ttl_seconds)

    def get_recommendation(self, investor_id: str, query: str) -> Optional[Dict]:
        query_hash = self._hash_query(investor_id, query)
        key = f"rec:{query_hash}"
        return self._get(key)

    # =========================================================================
    # STATS & MANAGEMENT
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
        
        cached_keys = 0
        if self.is_available():
            try:
                cached_keys = self.cache_col.count_documents({"expires_at": {"$gt": datetime.utcnow()}})
            except:
                cached_keys = len(self.local_fallback)
        else:
            cached_keys = len(self.local_fallback)
            
        return {
            "backend": "mongodb" if self.is_available() else "memory_fallback",
            "cached_keys": cached_keys,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4)
        }

    def clear(self) -> bool:
        self.local_fallback.clear()
        if self.is_available():
            try:
                self.cache_col.delete_many({})
                print("🗑️  MongoDB cache cleared")
                return True
            except Exception as e:
                print(f"❌ Failed to clear MongoDB cache: {e}")
                return False
        return True


# Backward-compatible alias
RedisCache = MongoCache

_cache_instance = None

def get_cache() -> MongoCache:
    """Get or create the global MongoCache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MongoCache()
    return _cache_instance

get_redis_cache = get_cache

def close_cache():
    """Close the global cache"""
    global _cache_instance
    if _cache_instance:
        _cache_instance.close()
        _cache_instance = None

close_redis_cache = close_cache
