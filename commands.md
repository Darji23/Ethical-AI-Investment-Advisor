# 🛠️ MongoDB Database Management Commands

This guide details commands to inspect, manage, seed, and reset MongoDB collections ("tables") for the Ethical AI Investment Advisor without requiring external shell tools like `mongosh`.

All commands use the built-in [scripts/manage_db.py](file:///Users/abhishekdarji/MyDrive/Ethical-AI-Investment-Advisor/scripts/manage_db.py) script and run inside your active virtual environment.

---

## 📋 Quick Reference Commands

### 1. Check Database Status & Collection Counts
Inspect active collections, document counts, and connectivity in your MongoDB cluster:
```bash
python3 -m scripts.manage_db status
```

### 2. Drop Audit Logs
Drop `advisor_logs`, `blocked_recommendations`, and `system_events` from MongoDB:
```bash
python3 -m scripts.manage_db drop-logs
```

### 3. Drop a Specific Collection
Drop any individual collection by name (e.g. `advisor_logs`, `products`, `cache`, etc.):
```bash
python3 -m scripts.manage_db drop advisor_logs
```

### 4. Clear Cache Collection
Purge the cached investor profiles and recommendations:
```bash
python3 -m scripts.manage_db drop-cache
```

### 5. Seed Investor Profiles & Products
Migrate datasets from `data/investor_profiles.json` and `data/product_master.json` into MongoDB collections:
```bash
python3 -m scripts.manage_db seed
```

### 6. Drop Entire Database
Drop all collections and reset the database (interactive prompt):
```bash
python3 -m scripts.manage_db drop-all
```

---

## 📡 Data Ingestion Commands

### 7. Full Ingestion (Products + Investors + FAISS)
Fetch ~270 real ETFs from Yahoo Finance, generate 5,000 synthetic investors, and rebuild the FAISS index:
```bash
python3 -m scripts.ingest_data
```

### 8. Products Only
Fetch real ETF data and calculate volatility-based risk scores:
```bash
python3 -m scripts.ingest_data --products
```

### 9. Investors Only
Generate 5,000 synthetic investor profiles (or custom count):
```bash
python3 -m scripts.ingest_data --investors
python3 -m scripts.ingest_data --investors --investor-count 10000
```

### 10. Rebuild FAISS Index Only
Re-embed product descriptions and rebuild the vector store:
```bash
python3 -m scripts.ingest_data --rebuild
```

### 11. Resume Interrupted Product Fetch
Resume from the last checkpoint if yfinance rate-limited your previous run:
```bash
python3 -m scripts.ingest_data --resume
```

---

## ⚡ Direct Python One-Liners (Alternative)

If you prefer direct one-line terminal commands:

#### Drop `advisor_logs`
```bash
python3 -c "from pymongo import MongoClient; c = MongoClient('mongodb://localhost:27017'); c['ethical_advisor']['advisor_logs'].drop(); c['audit_logs']['advisor_logs'].drop(); print('✅ advisor_logs dropped')"
```

#### Drop `cache`
```bash
python3 -c "from pymongo import MongoClient; c = MongoClient('mongodb://localhost:27017'); c['ethical_advisor']['cache'].drop(); print('✅ cache dropped')"
```

#### View Document Counts
```bash
python3 -c "from pymongo import MongoClient; db = MongoClient('mongodb://localhost:27017')['ethical_advisor']; print({col: db[col].count_documents({}) for col in db.list_collection_names()})"
```
