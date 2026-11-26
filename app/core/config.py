"""
Configuration variables and environment setups
All database storage, caching, and audit trails connect to MongoDB in the same cluster.
"""
import os
from pathlib import Path

# Automatically load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Lightweight fallback parser if python-dotenv is not installed
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val

# Port setup (HF Spaces use 7860, local default to 8000)
PORT = int(os.getenv("PORT", 8000))

# Jina Embedding Configuration
# Prime model: API-tier flagship "jina-embeddings-v3" (with task-specific LoRA adapters)
# Fallback model: Open-weights "jina-embeddings-v5-text-small" (or "jina-embeddings-v5-omni-small")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v3")
JINA_FALLBACK_MODEL = os.getenv("JINA_FALLBACK_MODEL", "jina-embeddings-v5-text-small")

# Fireworks AI Configuration
# Prime Model: State-of-the-art accuracy, ultra-fast & low-cost (deepseek-v4p1-flash)
# Fallback Model: Open-weights model (glm-5p3-flash)
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/deepseek-v4p1-flash")
FIREWORKS_FALLBACK_MODEL = os.getenv("FIREWORKS_FALLBACK_MODEL", "accounts/fireworks/models/glm-5p3-flash")

# MongoDB Unified Cluster Configuration (Collections for DB, Cache, and Audit logs)
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "ethical_advisor")
