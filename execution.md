# Execution Guide - Ethical AI Investment Advisor

This document contains step-by-step instructions on how to execute this project locally (using a python virtual environment) and deploy it to Hugging Face Spaces.

---

## 🔑 Environment Secrets Setup

For the RAG pipeline to generate real AI-powered advice, you require:
- **Jina AI API Key & Model**: Get one at [Jina AI](https://jina.ai/) (free tier available). Supports `jina-embeddings-v5-text` and `jina-embeddings-v5-omni`.
- **Fireworks AI API Key**: Get one at [Fireworks AI](https://fireworks.ai/) (free tier available).

Set these keys as environment variables in your terminal (or in `.env`):
```bash
export JINA_API_KEY="your_jina_key_here"
export JINA_MODEL="jina-embeddings-v5-text" # or "jina-embeddings-v5-omni"
export FIREWORKS_API_KEY="your_fireworks_key_here"
export FIREWORKS_MODEL="accounts/fireworks/models/kimi-k2p5"
```
*Note: If these keys are not set, the advisor automatically runs in **Demo Mode** using mock AI structures. Suitability checks, metrics, and audits remain 100% operational.*

---

## 💻 Option 1: Running Locally (Virtual Environment)

This option is the fastest way to test the application. It runs the FastAPI server with a robust local memory/JSON file fallback for database, cache, and audit tracking.

### Step 1: Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Populate MongoDB and Vector Store
Fetch real ETF market data, generate 5,000 synthetic investor profiles, and build the FAISS index:
```bash
# Ingest real ETF catalog + synthetic investor profiles + rebuild FAISS index
python3 -m scripts.ingest_data

# Or run individual steps:
python3 -m scripts.ingest_data --products    # Fetch real ETFs only
python3 -m scripts.ingest_data --investors   # Generate synthetic profiles only
python3 -m scripts.ingest_data --rebuild     # Rebuild FAISS index only
```

### Step 4: Run the FastAPI Server
```bash
python3 -m app.main
```
The server will start on **`http://localhost:8000`**. 

Open this URL in your web browser to access the interactive dashboard, view client profiles, submit investment queries, explore catalogs, and trigger live Fairness Audits!

---

## 🚀 Option 2: Deploying on Hugging Face Spaces

Hugging Face Spaces build the project using the included Dockerfile. By default, Hugging Face Docker Spaces bind to port `7860`.

### Step 1: Create a New Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click on **Spaces** -> **Create new Space**.
3. Choose a name (e.g. `ethical-ai-investment-advisor`).
4. Select **Docker** as the SDK.
5. Choose **Blank** (or standard template) as the Docker template.
6. Select the **Free CPU basic** hardware tier.
7. Click **Create Space**.

### Step 2: Configure Environment Keys
To let the space use real AI embeddings and models:
1. Go to your Space **Settings** tab.
2. Scroll to the **Variables and Secrets** section.
3. Click **New Secret** to add:
   - `JINA_API_KEY`: Value from your Jina portal.
   - `FIREWORKS_API_KEY`: Value from your Fireworks portal.

### Step 3: Push Your Code
Clone the space repository locally, copy the restructured files into it, commit, and push:
```bash
# Clone space
git clone https://huggingface.co/spaces/your-username/your-space-name
cd your-space-name

# Copy codebase contents to repository
cp -r /path/to/restructured/project/* .

# Commit and push
git add .
git commit -m "Initial commit of restructured advisor application"
git push
```
Hugging Face will automatically build the `Dockerfile`, spin up the container, and serve your glassmorphism interactive dashboard directly in the Space on port `7860`.

---

## 📊 Viewing Prometheus & Grafana Dashboards After Deployment

Because Hugging Face Spaces run as a single container and expose only one public web port (port `7860`), you cannot run Grafana (port `3000`) and Prometheus (port `9090`) side-by-side on separate ports inside the same Space container.

Instead, the production-grade way to monitor your Space is to **scrape the FastAPI metrics endpoint from a free Grafana Cloud account**.

### How it works:
FastAPI serves raw Prometheus metrics publicly at the **`/metrics`** endpoint (e.g., `https://your-username-your-space-name.hf.space/metrics`).

Follow these steps to set up Prometheus & Grafana in 5 minutes:

1. **Get Direct Space URL**:
   Your Hugging Face Space has a direct web URL (distinct from the HF iframe wrapper). It follows this format:
   `https://[username]-[space-name].hf.space` (e.g., `https://abhishekdarji-ethical-ai-advisor.hf.space`)
   Verify that you can access the metrics page by navigating to `https://[username]-[space-name].hf.space/metrics`.

2. **Sign Up for Grafana Cloud (100% Free)**:
   Go to [Grafana Cloud](https://grafana.com/products/cloud/) and sign up for a free account. The free tier includes Prometheus metrics storage and full Grafana dashboarding.

3. **Configure Grafana Cloud Prometheus to scrape your Space**:
   - In your Grafana Cloud dashboard, navigate to **Connections** -> **Add connection**.
   - Search and select **Hosted Prometheus** (or **Prometheus**).
   - In the scrape configuration, add your Space metrics endpoint as a static target:
     ```yaml
     scrape_configs:
       - job_name: 'hf-space-advisor'
         metrics_path: '/metrics'
         scheme: 'https'
         static_configs:
           - targets: ['[username]-[space-name].hf.space']
     ```
   - Click save. Grafana Cloud will now pull real-time metrics from your Hugging Face Space every 15s.

4. **Import the Grafana Dashboard**:
   - In Grafana Cloud, go to **Dashboards** -> **New** -> **Import**.
   - Open the `app/static/grafana_dashboard.json` file in this codebase (or download it directly from your deployed Hugging Face Space by going to `https://[username]-[space-name].hf.space/static/grafana_dashboard.json`).
   - Copy the entire contents of `grafana_dashboard.json` and paste it into the Grafana Import JSON text box.
   - Select your hosted Prometheus data source and click **Import**.
   - You will see live request rates, approval ratios, RAG latency, and suitability violations directly in your Grafana Cloud dashboard!

---

## 🧪 Validating the API
To test endpoints and verify validation triggers from your terminal, execute the test client:
```bash
python3 -m scripts.test_api
```
To run API performance benchmarks:
```bash
python3 -m scripts.benchmark
```
