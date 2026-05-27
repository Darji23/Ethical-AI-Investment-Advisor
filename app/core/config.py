"""
Configuration variables and environment setups
"""
import os

# Port setup (HF Spaces use 7860, local default to 8000)
PORT = int(os.getenv("PORT", 8000))

# API Keys
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")

# Databases URLs
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://advisor:advisor123@localhost:5432/advisor_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
