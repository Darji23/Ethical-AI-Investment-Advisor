# 🤖 Ethical AI Investment Advisor
## High-Performance RAG Advisory & Compliance System with Grafana Cloud Monitoring

A production-ready, regulation-compliant investment advisory system with advanced ethical guardrails, semantic search via RAG, a responsive glassmorphism UI, and real-time Prometheus telemetry designed for seamless deployment on **Hugging Face Spaces**.

---

## 🎯 **Core Objective**
This system provides **safe, personalized investment recommendations** with dual-layer compliance validation. All investor profiles, product catalogs, caches, and audit logs are managed natively in **MongoDB** across dedicated collections (`investor_profiles`, `products`, `cache`, `advisor_logs`, `blocked_recommendations`).

- ✅ **Ethical Guardrails** - 12 strict regulatory and demographic safety rules protecting investors.
- ✅ **Advanced RAG** - Semantic search powered by Jina Embeddings + Fireworks LLM (with offline Mock fallbacks).
- ✅ **Dynamic Auditing** - Real-time Demographic Parity, Parity Gaps, and Chi-Square audits computed dynamically.
- ✅ **Premium Frontend** - Stunning interactive glassmorphism dashboard built with raw CSS micro-animations.
- ✅ **Cloud Telemetry** - Exposed `/metrics` Prometheus endpoint ready to be scraped by Grafana Cloud.

---

## 📦 **Streamlined Project Structure**

The codebase has been restructured into a modular, clean Python application:

```
Ethical-AI-Investment-Advisor/
│
├── 📁 app/                     # Core Application Package
│   ├── 📁 core/                # Database, cache, and config settings (with fallbacks)
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── cache.py
│   │   └── mongodb_audit.py
│   │
│   ├── 📁 schemas/             # Pydantic structured data schemas
│   │   └── schemas.py
│   │
│   ├── 📁 services/            # RAG, Suitability, & Audit Engines
│   │   ├── suitability_validator.py
│   │   ├── fairness_audit.py
│   │   ├── jina_embedder.py
│   │   ├── fireworks_llm.py
│   │   ├── vector_store.py
│   │   ├── context_injector.py
│   │   └── rag_advisor.py
│   │
│   ├── 📁 static/              # Stunning Dashboard Frontend & Metrics Assets
│   │   ├── index.html          # Interactive dark-mode dashboard
│   │   ├── css/styles.css      # Harmonious HSL styling & micro-animations
│   │   ├── js/app.js           # Chart rendering & consultations engine
│   │   └── grafana_dashboard.json # Pre-built Grafana Cloud monitoring dashboard
│   │
│   └── main.py                 # FastAPI Entrypoint (Serves UI, API, & Prometheus metrics)
│
├── 📁 scripts/                 # System Utilities, Load Tests & Benchmarks
│   ├── ingest_data.py          # Real data ingestion pipeline (ETFs + investors + FAISS)
│   ├── mock_data.py            # Legacy mock data generator (kept for reference)
│   ├── vector_store.py         # Standalone FAISS vector store builder
│   ├── manage_db.py            # MongoDB management utility (status, seed, drop)
│   ├── benchmark.py            # Latency and Cache validation benchmarks
│   ├── locust_load_test.py     # Locust load simulation script
│   └── test_api.py             # HTTP terminal client verification
│
├── 📁 data/                    # Vector store index storage
│   ├── product_vector_store.index
│   └── product_vector_store.docs
│
├── Dockerfile                  # HF Space compatible single-container Dockerfile
├── requirements.txt            # Consolidated python dependencies and requirements
├── README.md                   # Project description and setup
└── execution.md                # Step-by-step execution & Grafana Cloud setup guide
```

---

## 🚀 **Quick Start (Local Execution)**

### **Step 1: Install Dependencies**
Create a standard virtual environment and install the required dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Step 2: Configure Environment Variables**
Copy the example environment file and add your API keys:
```bash
cp .env.example .env
# Edit .env to add your JINA_API_KEY, FIREWORKS_API_KEY, and MONGODB_URL
```

### **Step 3: Ingest Real Data**
Fetch 197 real ETF products from Yahoo Finance, generate 5,500 `synthetic-v2` investors, and build the FAISS vector index:
```bash
# Full ingestion pipeline (products + investors + FAISS rebuild)
python3 -m scripts.ingest_data

# Or run individual steps:
python3 -m scripts.ingest_data --products    # Fetch real ETFs only
python3 -m scripts.ingest_data --investors   # Generate synthetic profiles only
python3 -m scripts.ingest_data --rebuild     # Rebuild FAISS index only
python3 -m scripts.ingest_data --resume      # Resume interrupted product fetch
```

> **Note:** Product ingestion takes ~5 minutes (rate-limited Yahoo Finance API calls). Progress is saved incrementally — use `--resume` if interrupted.

### **Step 4: Run the FastAPI Server**
Start the application locally:
```bash
python3 -m app.main
```
Navigate to **`http://localhost:8000`** in your browser to interact with the dashboard!
*To verify that Prometheus metrics are live locally, navigate to `http://localhost:8000/metrics`.*

---

## ☁️ **Deploying to Hugging Face Spaces**

Hugging Face Spaces build the project seamlessly using the included `Dockerfile` and default to serving the app on port `7860`.

1. **Create the Space**:
   - Log in to [Hugging Face](https://huggingface.co/).
   - Click **Spaces** -> **Create new Space**.
   - Select **Docker** as the SDK and choose **Blank** as the template.
   - Click **Create Space**.

2. **Add Environment Secrets (Optional for full AI)**:
   - In your Space's **Settings** tab, go to **Variables and secrets**.
   - Add `JINA_API_KEY` and `FIREWORKS_API_KEY` for real AI advice generation.
   - *If no Fireworks key is added, recommendations come from a local open-weights Hugging Face model (default `Qwen2.5-1.5B-Instruct`; MLX 4-bit on Apple Silicon, `transformers` elsewhere). No API key is needed; weights download on first use. Configure with `LOCAL_LLM_BACKEND` / `LOCAL_LLM_MODEL`. The same model is the last-resort tier when both Fireworks models fail. Without a Jina key, embeddings still use the offline mock embedder.*

3. **Deploy Code**:
   - Clone your Hugging Face Space repository:
     ```bash
     git clone https://huggingface.co/spaces/your-username/your-space-name
     cd your-space-name
     ```
   - Copy the project files inside, then push:
     ```bash
     cp -r /path/to/Ethical-AI-Investment-Advisor/* .
     git add .
     git commit -m "Deploy restructured advisor codebase"
     git push
     ```
   Hugging Face will compile the image, start the container, and serve your dashboard directly!

---

## 📊 **Prometheus & Grafana Dashboards via Grafana Cloud**

Because Hugging Face Space runs as a single container exposing a single web port (`7860`), we expose Prometheus-formatted metrics at **`/metrics`** and scrap them using a free Grafana Cloud instance.

### **Quick Setup Steps**:

1. **Sign Up for Grafana Cloud**:
   Go to [Grafana Cloud](https://grafana.com/products/cloud/) and sign up for a **100% free account** (includes Prometheus storage).

2. **Configure Scrape Target**:
   Find your Hugging Face Space direct URL (e.g., `https://[username]-[space-name].hf.space`). In Grafana Cloud, add your space's `/metrics` as a hosted Prometheus scrape target:
   ```yaml
   scrape_configs:
     - job_name: 'hf-space-advisor'
       metrics_path: '/metrics'
       scheme: 'https'
       static_configs:
         - targets: ['[username]-[space-name].hf.space']
   ```

3. **Import Dashboard**:
   - In Grafana Cloud, navigate to **Dashboards** -> **New** -> **Import**.
   - Open [grafana_dashboard.json](app/static/grafana_dashboard.json) (or download it from `https://your-space.hf.space/static/grafana_dashboard.json`).
   - Copy-paste the JSON file contents and click **Import**.

You are all set! You will see live request rates, P95 processing times, approval vs rejection curves, and compliance rule violations in your browser.

---

## 🧪 **Testing & Benchmarks**

Run validations to ensure the API, RAG, and safety layers work optimally:
```bash
# Verify all endpoints and triggers
python3 -m scripts.test_api

# Run local load performance benchmarks
python3 -m scripts.benchmark

# Run Locust load simulations (configure at http://localhost:8089)
locust -f scripts/locust_load_test.py
```

---

## 📈 **Data Pipeline Architecture**

### Real ETF Products (197 active products, Catalog `catalog_v2`)
- **Source:** Yahoo Finance via `yfinance` — real fund names, descriptions, expense ratios, share prices, and 5-year daily prices.
- **Risk Scoring Method:** `VOL-5band-v1 (SRRI-inspired)` scale calculated from annualized historical daily volatility:
  - Risk 1 (Conservative): $\sigma < 5\%$ — money markets, short-term treasuries (`SHV`, `BIL`)
  - Risk 2 (Mod-Conservative): $5\% \le \sigma < 10\%$ — aggregate bonds, investment-grade corporate (`BND`, `AGG`, `GOVT`)
  - Risk 3 (Moderate): $10\% \le \sigma < 20\%$ — broad equity (S&P 500, total market) (`SPY`, `VOO`, `VTI`)
  - Risk 4 (Mod-Aggressive): $20\% \le \sigma < 40\%$ — tech, semiconductors, clean energy (`QQQ`, `SOXX`, `ARKK`)
  - Risk 5 (Aggressive): $\sigma \ge 40\%$ **or** any leveraged/inverse fund (`TQQQ`, `SQQQ`, `SPXL`, `SOXL`)
- **Leverage Detection:** Category-first detection (`Trading--Leveraged/Inverse Equity/Debt`) with regex backup (`2X`, `3X`, `-1X`, `-2X`, `-3X`). Disambiguates ProShares Ultra/UltraPro ($2\times, 3\times$) from non-leveraged Ultra-Short Bond ETFs.
- **Asset Classes (Holdings-Based):** `equity`, `fixed_income`, `commodity`, `real_estate`, `alternative` (crypto/digital assets), and `cash` (<1 yr maturity).
- **Search Quality:** Full fund mandates from Yahoo Finance embedded using Jina AI (`jina-embeddings-v3`, 1024-dim) into FAISS with model provenance tracking.

### Synthetic Investor Profiles (5,500 profiles, `synthetic-v2`)
- **IDs & Provenance:** Zero-padded `INV00001`–`INV05500` with `_generator: "synthetic-v2"`, `_seed: 42`, and unique MongoDB index on `investor_id`.
- **Schema Integrity vs. Suitability:** The Pydantic v2 schema validates data integrity only (ranges, `investable_assets <= net_worth`, valid `investment_objective` and `liquidity_needs`), leaving all suitability judgments (including age-risk limits) to `SuitabilityValidator`.
- **Distributional Assumptions:** All generator parameters are synthetic assumptions (not sampled directly from real data). Income uses a right-skewed lognormal distribution loosely inspired by the U.S. Federal Reserve Survey of Consumer Finances (SCF 2022); investable assets grow noisily with age and income; time horizon shrinks with age; experience grows with age; and risk tolerance trends lower with age and higher with experience—with noise so unusual combinations still appear.
- **Deliberate Edge Cases (`segment: "edge_case"`):** 10 hard-coded boundary profiles (`INV00001`–`INV00010`), including a 75-year-old with risk tolerance 5, a 22-year-old beginner with \$2,000 investable assets, a 1-year horizon seeking speculation, high income with minimal investable assets, and exact Rule 4 age boundaries (64, 65, 69, 70).
- **Contradiction Detection:** Profiles with conflicting attributes (e.g., `risk_tolerance=1` with `speculation`, or `risk_tolerance=5` with `time_horizon<=2`) are flagged in `profile_warnings`.
- **Counterfactual Fairness:** Investor `name` is excluded from all LLM prompts so demographic proxies cannot influence recommendations.

---

**Built with:** Python | FastAPI | Hugging Face Spaces | Jina AI | Fireworks AI | FAISS | MongoDB | yfinance | Prometheus | Grafana Cloud
