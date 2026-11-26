"""
Cache Layer - Re-export MongoDB Cache
Provides backward compatibility for modules importing from app.core.redis_cache
"""
from app.core.cache import (
    MongoCache,
    RedisCache,
    get_cache,
    get_redis_cache,
    close_cache,
    close_redis_cache
)

__all__ = [
    "MongoCache",
    "RedisCache",
    "get_cache",
    "get_redis_cache",
    "close_cache",
    "close_redis_cache"
]
