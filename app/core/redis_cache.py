"""
Redis Cache Manager - Actually Use Redis for Caching!
Provides caching for investors, products, and recommendations
"""
import redis
import json
import os
from typing import Optional, List, Dict, Any

from app.schemas.schemas import InvestorProfile, Product
from app.core.config import REDIS_URL

class RedisCache:
    """Redis cache manager with proper TTL and key management"""
    
    def __init__(self, redis_url: str = None):
        """
        Initialize Redis connection
        """
        if redis_url is None:
            redis_url = REDIS_URL
        
        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            self.client.ping()
            self.connected = True
            print("✅ Redis cache connected and ready")
        except Exception as e:
            print(f"⚠️  Redis not available: {e}")
            self.client = None
            self.connected = False
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self.connected and self.client is not None
    
    # =========================================================================
    # INVESTOR PROFILE CACHING
    # =========================================================================
    
    def cache_investor(self, investor: InvestorProfile, ttl_seconds: int = 3600):
        """
        Cache an investor profile
        """
        if not self.is_available():
            return False
        
        try:
            key = f"investor:{investor.investor_id}"
            value = json.dumps(investor.dict())
            self.client.setex(key, ttl_seconds, value)
            return True
        except Exception as e:
            print(f"❌ Failed to cache investor: {e}")
            return False
    
    def get_investor(self, investor_id: str) -> Optional[InvestorProfile]:
        """
        Get cached investor profile
        """
        if not self.is_available():
            return None
        
        try:
            key = f"investor:{investor_id}"
            cached = self.client.get(key)
            
            if cached:
                return InvestorProfile(**json.loads(cached))
            return None
        except Exception as e:
            print(f"❌ Failed to get cached investor: {e}")
            return None
    
    def cache_all_investors(self, investors: List[InvestorProfile], ttl_seconds: int = 3600):
        """
        Cache multiple investors at once (bulk operation)
        """
        if not self.is_available():
            return 0
        
        count = 0
        for investor in investors:
            if self.cache_investor(investor, ttl_seconds):
                count += 1
        return count
    
    # =========================================================================
    # PRODUCT CACHING
    # =========================================================================
    
    def cache_product(self, product: Product, ttl_seconds: int = 7200):
        """
        Cache a product
        """
        if not self.is_available():
            return False
        
        try:
            key = f"product:{product.ticker}"
            value = json.dumps(product.dict())
            self.client.setex(key, ttl_seconds, value)
            return True
        except Exception as e:
            print(f"❌ Failed to cache product: {e}")
            return False
    
    def get_product(self, ticker: str) -> Optional[Product]:
        """
        Get cached product
        """
        if not self.is_available():
            return None
        
        try:
            key = f"product:{ticker}"
            cached = self.client.get(key)
            
            if cached:
                return Product(**json.loads(cached))
            return None
        except Exception as e:
            print(f"❌ Failed to get cached product: {e}")
            return None
    
    def cache_all_products(self, products: List[Product], ttl_seconds: int = 7200):
        """
        Cache all products (bulk operation)
        """
        if not self.is_available():
            return 0
        
        count = 0
        for product in products:
            if self.cache_product(product, ttl_seconds):
                count += 1
        return count
    
    def get_all_products_list(self) -> Optional[str]:
        """
        Get the list of all product tickers from cache
        """
        if not self.is_available():
            return None
        
        try:
            return self.client.get("products:all_tickers")
        except:
            return None
    
    def cache_all_products_list(self, tickers: List[str], ttl_seconds: int = 7200):
        """
        Cache the list of all product tickers
        """
        if not self.is_available():
            return False
        
        try:
            self.client.setex("products:all_tickers", ttl_seconds, json.dumps(tickers))
            return True
        except:
            return False
    
    # =========================================================================
    # RECOMMENDATION CACHING
    # =========================================================================
    
    def cache_recommendation(
        self,
        investor_id: str,
        query_hash: str,
        recommendation: Dict,
        ttl_seconds: int = 1800
    ):
        """
        Cache a recommendation
        """
        if not self.is_available():
            return False
        
        try:
            key = f"recommendation:{investor_id}:{query_hash}"
            value = json.dumps(recommendation)
            self.client.setex(key, ttl_seconds, value)
            return True
        except Exception as e:
            print(f"❌ Failed to cache recommendation: {e}")
            return False
    
    def get_recommendation(self, investor_id: str, query_hash: str) -> Optional[Dict]:
        """
        Get cached recommendation
        """
        if not self.is_available():
            return None
        
        try:
            key = f"recommendation:{investor_id}:{query_hash}"
            cached = self.client.get(key)
            
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            print(f"❌ Failed to get cached recommendation: {e}")
            return None
    
    # =========================================================================
    # STATISTICS & MONITORING
    # =========================================================================
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get Redis cache statistics
        """
        if not self.is_available():
            return {
                "connected": False,
                "total_keys": 0,
                "memory_used": 0
            }
        
        try:
            info = self.client.info()
            investor_keys = len(self.client.keys("investor:*"))
            product_keys = len(self.client.keys("product:*"))
            recommendation_keys = len(self.client.keys("recommendation:*"))
            
            return {
                "connected": True,
                "total_keys": info.get('db0', {}).get('keys', 0),
                "investor_keys": investor_keys,
                "product_keys": product_keys,
                "recommendation_keys": recommendation_keys,
                "memory_used_bytes": info.get('used_memory', 0),
                "memory_used_human": info.get('used_memory_human', '0B'),
                "hit_rate": self._calculate_hit_rate(info)
            }
        except Exception as e:
            print(f"❌ Failed to get cache stats: {e}")
            return {"connected": False, "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate from Redis stats"""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        if total == 0:
            return 0.0
        return hits / total
    
    def clear_investor_cache(self, investor_id: str = None):
        """
        Clear investor cache (all or specific)
        """
        if not self.is_available():
            return 0
        
        try:
            if investor_id:
                key = f"investor:{investor_id}"
                return self.client.delete(key)
            else:
                keys = self.client.keys("investor:*")
                if keys:
                    return self.client.delete(*keys)
                return 0
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
            return 0
    
    def clear_all_cache(self):
        """Clear entire Redis cache"""
        if not self.is_available():
            return False
        
        try:
            self.client.flushdb()
            print("🗑️  Redis cache cleared")
            return True
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
            return False
    
    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            print("✅ Redis connection closed")

_redis_cache = None

def get_redis_cache() -> RedisCache:
    """Get or create the global Redis cache instance"""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache

def close_redis_cache():
    """Close the global Redis cache"""
    global _redis_cache
    if _redis_cache:
        _redis_cache.close()
        _redis_cache = None
