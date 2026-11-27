# Commands

Every command runs from the project root, inside the virtual environment.

- [1. Setup](#1-setup)
- [2. Data ingestion](#2-data-ingestion)
- [3. Run the app](#3-run-the-app)
- [4. Tests and checks](#4-tests-and-checks)
- [5. Benchmark and fairness test](#5-benchmark-and-fairness-test)
- [6. Monitoring with Prometheus and Grafana](#6-monitoring-with-prometheus-and-grafana)
- [7. Database management](#7-database-management)

---

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional: the local open-weights LLM tier (used only when no Fireworks key is set, or when both
Fireworks models fail):

```bash
pip install -r requirements-local-llm.txt
```

Create your environment file and fill in the keys:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `JINA_API_KEY` | Embeddings ([jina.ai](https://jina.ai/)) |
| `FIREWORKS_API_KEY` | LLM ([fireworks.ai](https://fireworks.ai/)) |
| `MONGODB_URL` | MongoDB connection, default `mongodb://localhost:27017` |
| `MONGODB_DB_NAME` | Database name, default `ethical_advisor` |

`.env` is loaded automatically. Without the API keys, the app falls back to mock components:
the suitability rules still work, but recommendations are not real.

Start MongoDB locally if it is not already running:

```bash
brew services start mongodb-community
```

---

## 2. Data ingestion

```bash
# Everything: ETFs + investors + FAISS index
python3 -m scripts.ingest_data

# Individual steps
python3 -m scripts.ingest_data --products                          # fetch ETFs from Yahoo Finance
python3 -m scripts.ingest_data --investors                         # 5,500 synthetic investors
python3 -m scripts.ingest_data --investors --investor-count 10000  # custom count
python3 -m scripts.ingest_data --rebuild                           # rebuild the FAISS index only
python3 -m scripts.ingest_data --resume                            # resume an interrupted product fetch
```

Product ingestion takes about 5 minutes because Yahoo Finance is rate-limited.
Rebuild the FAISS index whenever products or the embedding model change.

---

## 3. Run the app

```bash
python3 -m app.main
# or
uvicorn app.main:app --reload
```

| URL | What it is |
|---|---|
| http://localhost:8000 | Web dashboard |
| http://localhost:8000/docs | Interactive API docs |
| http://localhost:8000/metrics | Prometheus metrics |
| http://localhost:8000/health | Health check |

Main endpoints: `POST /consult`, `GET /investors`, `GET /investors/{investor_id}`,
`GET /products`, `POST /audit/simulate`, `GET /audit/stats`.

---

## 4. Tests and checks

```bash
# Unit tests: every rule boundary and failure path
python3 -m pytest -q tests/

# Call every API endpoint (app must be running)
python3 -m scripts.test_api

# Latency benchmark (app must be running)
python3 -m scripts.benchmark

# Load test (app must be running), then open http://localhost:8089
locust -f scripts/locust_load_test.py --host=http://localhost:8000

# Check which Fireworks models your key can reach
python3 -m scripts.diagnose_models
```

---

## 5. Benchmark and fairness test

### 3-version violation-rate benchmark

Compares the raw LLM, rules in the prompt only, and the full system on the same cases.

```bash
# Quick smoke test
SAMPLE_SIZE=10 python3 -m scripts.measure_violation_rate

# Full run (about 75 minutes for 200 cases)
caffeinate -i &        # macOS: keep the laptop awake
SAMPLE_SIZE=200 python3 -m scripts.measure_violation_rate
```

Outputs in `data/`:

| File | Contents |
|---|---|
| `benchmark_comparison.txt` | Summary tables |
| `violation_rate_report.json` | Every case, answer and outcome |
| `benchmark_all_answers.txt` | Readable per-case log |
| `benchmark_comparison.png` | Comparison chart |

While it runs, the benchmark exposes its own metrics on **:8001** (see section 6).

Quick check of which model produced each answer:

```bash
grep -o '"model_tier": "[^"]*"' data/violation_rate_report.json | sort | uniq -c
```

### Counterfactual fairness test

Sends the same profiles with different names (plus repeated nameless runs as a noise baseline)
and measures how often the recommendation changes.

```bash
python3 -m scripts.counterfactual_fairness_test
```

---

## 6. Monitoring with Prometheus and Grafana

`prometheus.yml` scrapes the app (`localhost:8000`) and the benchmark (`localhost:8001`) every 5 seconds.

### Start Prometheus and Grafana (Homebrew)

```bash
brew install prometheus grafana     # once

# Terminal 1: Prometheus. Keep its storage outside the repo.
prometheus --config.file=prometheus.yml --storage.tsdb.path=$HOME/prometheus-data

# Grafana runs as a background service
brew services start grafana
```

### Connect Grafana (once)

1. Open http://localhost:3000 and log in with `admin` / `admin`.
2. **Connections → Data sources → Add data source → Prometheus**, URL `http://localhost:9090`, then **Save & test**.
3. **Dashboards → New → Import**, upload `app/static/grafana_dashboard.json`, and choose the Prometheus data source.

### Check the targets

Open http://localhost:9090/targets:

- `localhost:8000` is **UP** while the app is running.
- `localhost:8001` is **UP** only while the benchmark is running (it stays up 30 seconds after it finishes).

### Useful queries

| Panel | Query |
|---|---|
| Benchmark progress | `benchmark_progress_ratio` |
| Violation rate by version | `sum by (arm) (benchmark_violating_cases_total) / sum by (arm) (benchmark_cases_total)` |
| Outcomes by version | `sum by (arm, outcome) (benchmark_outcomes_total)` |
| Violations by rule | `sum by (arm, rule) (benchmark_rule_violations_total)` |
| Average time per case | `sum(benchmark_case_seconds_sum) / sum(benchmark_case_seconds_count)` |
| App request rate | `sum by (endpoint) (rate(http_requests_total[1m]))` |
| App p95 latency | `histogram_quantile(0.95, sum by (le, endpoint) (rate(http_request_duration_seconds_bucket[5m])))` |
| Guardrail interventions | `sum by (rule_id) (guardrail_interventions_total)` |
| Recovery paths | `sum by (path) (advisor_recovery_total)` |

### Stop

```bash
# Prometheus: Ctrl+C in its terminal
brew services stop grafana
```

---

## 7. Database management

```bash
python3 -m scripts.manage_db status              # collections and document counts
python3 -m scripts.manage_db drop-logs           # drop audit logs
python3 -m scripts.manage_db drop-cache          # clear the cache collection
python3 -m scripts.manage_db drop advisor_logs   # drop one collection by name
python3 -m scripts.manage_db drop-all            # drop everything (asks for confirmation)
```

Document counts in one line:

```bash
python3 -c "from pymongo import MongoClient; db = MongoClient('mongodb://localhost:27017')['ethical_advisor']; print({c: db[c].count_documents({}) for c in db.list_collection_names()})"
```

Check the investor data is clean (5,500 profiles, all `synthetic-v2`):

```bash
python3 -c "from pymongo import MongoClient; db = MongoClient('mongodb://localhost:27017')['ethical_advisor']; print(db.investor_profiles.count_documents({}), db.investor_profiles.distinct('_generator'))"
```