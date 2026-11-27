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
# JSON mode (response_format={"type": "json_object"}) for every Fireworks call; auto-disabled per model if Fireworks rejects it
FIREWORKS_JSON_MODE = os.getenv("FIREWORKS_JSON_MODE", "true").lower() in ("1", "true", "yes")

# Number of products retrieved from FAISS per request (after the Tier 1 pre-filter).
# Shared by the website (/consult) and scripts/measure_violation_rate.py so the demo runs the measured system.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))

# Local open-weights LLM (Hugging Face Hub, no API key) — last-resort tier, or the only tier without FIREWORKS_API_KEY
# Backend: auto | mlx (Apple Silicon) | transformers (any platform, needs torch)
LOCAL_LLM_BACKEND = os.getenv("LOCAL_LLM_BACKEND", "auto")
# Blank = backend default (mlx-community/Qwen2.5-1.5B-Instruct-4bit or Qwen/Qwen2.5-1.5B-Instruct)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "")
LOCAL_LLM_MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "768"))

# MongoDB Unified Cluster Configuration (Collections for DB, Cache, and Audit logs)
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "ethical_advisor")
