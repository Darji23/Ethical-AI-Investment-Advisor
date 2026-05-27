# 🤖 Ethical AI Investment Advisor
## High-Performance RAG Advisory & Compliance System with Grafana Cloud Monitoring

A production-ready, regulation-compliant investment advisory system with advanced ethical guardrails, semantic search via RAG, a responsive glassmorphism UI, and real-time Prometheus telemetry designed for seamless deployment on **Hugging Face Spaces**.

---

## 🎯 **Core Objective**
This system provides **safe, personalized investment recommendations** with dual-layer compliance validation. When deployed, it runs in **single-container mode**, using highly-optimized, automated in-memory and local JSON file fallbacks for database connection, caching, and audit logging. This ensures it is 100% operational out-of-the-box on Hugging Face without external dependencies, while still offering complete production readiness for external cloud connections.

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
│   │   ├── redis_cache.py
│   │   └── mongodb_audit.py
│   │
│   ├── 📁 schemas/             # Pydantic structured data schemas
│   │   └── schemas.py
│   │
│   ├── 📁 services/            # RAG, Suitability, & Audit Engines
│   │   ├── suitability_validator.py
│   │   ├── suitability_validator_expanded.py
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
│   ├── mock_data.py            # Compiles 500 investor profiles & product seeds
│   ├── vector_store.py         # Compiles FAISS vector store database
│   ├── benchmark.py            # Latency and Cache validation benchmarks
│   ├── locust_load_test.py     # Locust load simulation script
│   └── test_api.py             # HTTP terminal client verification
│
├── 📁 data/                    # Database seeds and FAISS index storage
│   ├── investor_profiles.json  # 500 randomized investor profiles
│   ├── product_master.json     # 25 catalog investment products
│   ├── product_vector_store.index
│   └── product_vector_store.docs
│
├── Dockerfile                  # HF Space compatible single-container Dockerfile
├── requirements.txt            # Consolidated, version-pinned python requirements
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

### **Step 2: Compile Vector Stores & Mock Data**
Generate 500 investor profiles and compile the local FAISS index:
```bash
# Generate seed datasets
python3 -m scripts.mock_data

# Create vector embeddings and indices
python3 -m scripts.vector_store
```

### **Step 3: Run the FastAPI Server**
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
   - *If no keys are added, the system runs automatically in Demo Mode using fast, offline mock embedders and mock LLM models.*

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
   - Open [grafana_dashboard.json](file:///Users/abhishekdarji/MyDrive/Ethical-AI-Investment-Advisor/app/static/grafana_dashboard.json) (or download it from `https://your-space.hf.space/static/grafana_dashboard.json`).
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

**Built with:** Python | FastAPI | Hugging Face Spaces | Jina AI | Fireworks AI | FAISS | Prometheus | Grafana Cloud
