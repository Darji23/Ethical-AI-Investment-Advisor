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
Fetch 197 real ETFs from Yahoo Finance, generate 5,500 `synthetic-v2` investors, and rebuild the FAISS index:
```bash
python3 -m scripts.ingest_data
```

### 8. Products Only
Fetch real ETF data and calculate volatility-based risk scores:
```bash
python3 -m scripts.ingest_data --products
```

### 9. Investors Only
Generate 5,500 `synthetic-v2` investor profiles (or custom count):
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

---

## 📊 Real-Time Monitoring — Benchmark as Live Traffic

Run the 3-arm benchmark (`measure_violation_rate.py`) while Prometheus scrapes it every 5s and Grafana displays the results live.

> **Files already in the repo:**
> - `prometheus.yml` — scrapes `:8000` (FastAPI) and `:8001` (benchmark)
> - `app/static/grafana_dashboard.json` — full dashboard with benchmark panels

---

### Step 1 — Start Prometheus + Grafana via Docker *(one-time)*

```bash
cd /Users/abhishekdarji/MyDrive/Ethical-AI-Investment-Advisor

docker network create monitoring

docker run -d --name prometheus --network monitoring -p 9090:9090 \
  -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml" \
  prom/prometheus

docker run -d --name grafana --network monitoring -p 3000:3000 grafana/grafana
```

### Step 2 — Connect Grafana to Prometheus *(browser, one-time)*

1. Open **http://localhost:3000** — login: `admin` / `admin`
2. **Connections → Data sources → Add → Prometheus**
3. URL: `http://prometheus:9090` → **Save & Test** (expect ✅ success)
4. **Dashboards → New → Import** → upload `app/static/grafana_dashboard.json`
5. When prompted for data source → pick the Prometheus you just added → **Import**

### Step 3 — Start the FastAPI app *(Terminal A)*

```bash
uvicorn app.main:app --reload
```

### Step 4 — Run the benchmark *(Terminal B)*

```bash
python3 -m scripts.measure_violation_rate
```

This automatically starts a Prometheus metrics server on **:8001** and keeps it alive for 30s after the benchmark finishes so Prometheus can collect the final numbers.

### Step 5 — Watch live in Grafana

Open **http://localhost:3000** → your imported dashboard.
Set time range to **Last 5 minutes**, auto-refresh every **5s** (pre-configured).

| Panel | What you see |
|-------|-------------|
| **Benchmark Progress** | 0 % → 100 % as cases complete |
| **Violation Rate by Arm** | 3 diverging lines — Raw high, Full ≈ 0 % |
| **Rules Broken — Prompt-Only** | Bar chart filling in per rule (R1–R7) |
| **Full System — Usable Rate** | Running fraction of usable portfolios |
| **Full System — Fallback Rate** | How often the safe fallback triggered |
| **P95 Case Duration** | Latency per case (3 LLM calls each) |

### Verify Prometheus targets *(optional)*

Open **http://localhost:9090/targets**

| Target | Expected state |
|--------|---------------|
| `:8000` (FastAPI) | ✅ UP (once app is running) |
| `:8001` (Benchmark) | ❌ DOWN until benchmark starts, then ✅ UP |

### Step 6 — Stop Docker when done

```bash
docker stop prometheus grafana && docker rm prometheus grafana
```
