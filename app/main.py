"""
FastAPI Server for Ethical AI Investment Advisor (Phase 4 Combined)
Serves static dashboard files, health checks, RAG consultation, catalog products,
and dynamic Fairness Audit simulation checks.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import json
import os
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, REGISTRY

# Core app imports
from app.schemas.schemas import InvestorProfile, PortfolioRecommendation, SuitabilityResult, Product
from app.services.vector_store import VectorStore
from app.services.suitability_validator import SuitabilityValidator
from app.services.rag_advisor import RAGInvestmentAdvisor
from app.services.fairness_audit import FairnessAuditor, generate_diverse_test_recommendations

# Database & Cache & Audit layers
from app.core.database import DatabaseManager
from app.core.redis_cache import RedisCache
from app.core.mongodb_audit import MongoAuditLogger
from app.core.config import PORT

# Initialize FastAPI App
app = FastAPI(
    title="Ethical AI Investment Advisor",
    description="Dual-Layer regulation-compliant advisory RAG system",
    version="4.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files (CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Global variables
advisor = None
validator = None
db_manager = None
redis_cache = None
mongo_logger = None

# Fallback local caches
investor_profiles_fallback = {}
products_fallback = []
audit_log_fallback = []

performance_stats = {
    'total_requests': 0,
    'total_approved': 0,
    'total_rejected': 0,
    'avg_response_time': 0
}

# Unregister custom metrics if they already exist in the registry (prevents errors on uvicorn reload)
for metric_name in ["http_requests_total", "http_request_duration_seconds", "advisor_decisions_total", "suitability_violations_total"]:
    if metric_name in REGISTRY._names_to_collectors:
        try:
            REGISTRY.unregister(REGISTRY._names_to_collectors[metric_name])
        except KeyError:
            pass

# Prometheus metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the FastAPI application",
    ["method", "endpoint", "status_code"]
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)

ADVISOR_DECISIONS_TOTAL = Counter(
    "advisor_decisions_total",
    "Total investment recommendations processed, labeled by final status (APPROVED/REJECTED)",
    ["status"]
)

SUITABILITY_VIOLATIONS_TOTAL = Counter(
    "suitability_violations_total",
    "Total number of suitability rule violations, labeled by the violated rule description",
    ["rule"]
)

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize all backend components with fallback architectures"""
    global advisor, validator, db_manager, redis_cache, mongo_logger, investor_profiles_fallback, products_fallback
    
    print("=" * 80)
    print("🚀 Starting Ethical AI Advisor API - Restructured & Robust")
    print("=" * 80)
    
    # 1. Connect database (PostgreSQL)
    print("📊 Connecting to PostgreSQL...")
    db_manager = DatabaseManager()
    
    db_investors = []
    db_products = []
    
    if db_manager.pool is not None:
        try:
            db_investors = db_manager.get_all_investors()
            db_products = db_manager.get_all_products()
            print(f"   PostgreSQL Connected: Found {len(db_investors)} investors, {len(db_products)} products.")
            
            # If database is empty, seed it
            if len(db_investors) == 0 or len(db_products) == 0:
                print("   Database is empty. Migrating seed files...")
                db_manager.migrate_from_json()
                db_investors = db_manager.get_all_investors()
                db_products = db_manager.get_all_products()
        except Exception as e:
            print(f"⚠️  Database query failed: {e}. Disabling pool.")
            db_manager.pool = None

    # 2. Connect Redis cache
    print("💾 Connecting to Redis...")
    redis_cache = RedisCache()
    
    # 3. Connect MongoDB audit logs
    print("📋 Connecting to MongoDB...")
    mongo_logger = MongoAuditLogger()
    
    # 4. Fallback: Load JSON datasets if DB is unavailable
    if not db_manager.pool:
        print("\n⚠️  PostgreSQL unavailable. Loading datasets from local JSON files...")
    
    try:
        # Load local products catalog
        with open('data/product_master.json', 'r') as f:
            products_data = json.load(f)
            products_fallback = [Product(**p) for p in products_data]
            print(f"   Loaded {len(products_fallback)} products from product_master.json")
            
        # Load local investor profiles
        with open('data/investor_profiles.json', 'r') as f:
            profiles_data = json.load(f)
            for profile_dict in profiles_data:
                investor_profiles_fallback[profile_dict['investor_id']] = InvestorProfile(**profile_dict)
            print(f"   Loaded {len(investor_profiles_fallback)} investors from investor_profiles.json")
    except Exception as e:
        print(f"❌ Failed to load fallback JSON datasets: {e}")
        # Generate them if they do not exist
        print("   Triggering mock data generator...")
        from scripts.mock_data import save_to_json
        save_to_json()
        with open('data/product_master.json', 'r') as f:
            products_fallback = [Product(**p) for p in json.load(f)]
        with open('data/investor_profiles.json', 'r') as f:
            for p in json.load(f):
                investor_profiles_fallback[p['investor_id']] = InvestorProfile(**p)

    # 5. Load or Compile Vector Store Index
    vector_store_path = 'data/product_vector_store'
    print("📚 Loading vector database...")
    
    if not (os.path.exists(f"{vector_store_path}.index") and os.path.exists(f"{vector_store_path}.docs")):
        print("⚠️  FAISS index not found. Compiling vector database...")
        from scripts.vector_store import build_vector_store
        build_vector_store()
        
    vector_store = VectorStore(embedding_dim=1024)
    vector_store.load(vector_store_path)
    
    # 6. Initialize Validator and Advisor
    active_products = db_products if db_manager.pool else products_fallback
    validator = SuitabilityValidator(active_products)
    
    # RAG advisor with fallbacks enabled
    advisor = RAGInvestmentAdvisor(
        vector_store=vector_store,
        validator=validator,
        use_real_ai=True  # Will automatically fall back to mock services if keys are missing
    )
    
    print("=" * 80)
    print("✅ BACKEND INITIALIZATION COMPLETE")
    print(f"   Listening on port: {PORT}")
    print("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup connections on shutdown"""
    print("👋 Shutting down advisor service...")
    if db_manager:
        db_manager.close_all()
    if redis_cache:
        redis_cache.close()
    if mongo_logger:
        mongo_logger.close()
    print("✅ Shutdown complete")

# ============================================================================
# API ROUTERS & CONTROLLERS
# ============================================================================

# Middleware to capture Prometheus request metrics
@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Exclude /metrics endpoint to avoid scrape noise
    path = request.url.path
    if path != "/metrics":
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=path,
            status_code=str(response.status_code)
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=path
        ).observe(duration)
        
    return response

class ConsultRequest(BaseModel):
    investor_id: str = Field(..., description="Unique investor profile ID")
    query: str = Field(..., min_length=10, description="Investment query directive")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the main interactive dashboard UI"""
    try:
        with open("app/static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")

@app.get("/health")
async def health_check():
    """Health check with component status tracking"""
    db_status = "healthy" if db_manager and db_manager.pool else "unavailable"
    redis_status = "healthy" if redis_cache and redis_cache.is_available() else "unavailable"
    mongo_status = "healthy" if mongo_logger and mongo_logger.is_available() else "unavailable"
    
    return {
        "status": "healthy",
        "service": "ethical-ai-advisor",
        "version": "4.1.0",
        "components": {
            "postgresql": db_status,
            "redis_cache": redis_status,
            "mongodb_audit": mongo_status,
            "rag_advisor": "operational" if advisor else "uninitialized"
        },
        "performance": performance_stats,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Exposes Prometheus-formatted system and application metrics"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ============================================================================
# MAIN ADVISORY PIPELINE
# ============================================================================

@app.post("/consult")
async def get_advice(request: ConsultRequest, background_tasks: BackgroundTasks):
    """
    Submit client advisory queries to RAG pipeline with dual-layer compliance validation.
    Supports multi-tier cache.
    """
    if not advisor:
        raise HTTPException(status_code=503, detail="RAG engine uninitialized.")
        
    start_time = time.time()
    
    # 1. Lookup profile (Redis -> DB -> Local JSON)
    investor_profile = None
    data_source = "json_file"
    
    # Tier 1: Redis cache
    if redis_cache and redis_cache.is_available():
        investor_profile = redis_cache.get_investor(request.investor_id)
        if investor_profile:
            data_source = "redis_cache"
            
    # Tier 2: PostgreSQL
    if not investor_profile and db_manager and db_manager.pool:
        investor_profile = db_manager.get_investor_profile(request.investor_id)
        if investor_profile:
            data_source = "postgresql"
            # Populate cache
            if redis_cache and redis_cache.is_available():
                redis_cache.cache_investor(investor_profile)
                
    # Tier 3: Local JSON fallback
    if not investor_profile:
        investor_profile = investor_profiles_fallback.get(request.investor_id)
        if investor_profile:
            data_source = "json_file"
            
    if not investor_profile:
        raise HTTPException(status_code=404, detail=f"Client profile {request.investor_id} not found.")

    try:
        # 2. Process query in RAG Advisor
        result = advisor.get_advice(
            investor_profile=investor_profile,
            user_query=request.query,
            top_k=5
        )
        
        end_time = time.time()
        total_time_ms = round((end_time - start_time) * 1000, 2)
        
        # 3. Compile timing metrics
        performance_metrics = {
            "total_time_ms": total_time_ms,
            "data_lookup_ms": round((time.time() - start_time) * 1000, 2) - total_time_ms, # approximate lookup delta
            "rag_pipeline_ms": total_time_ms # full RAG delta
        }
        
        validation = result['validation_result']
        recommendation = result['llm_recommendation']
        status = result['final_status']
        
        # Increment Prometheus metrics
        ADVISOR_DECISIONS_TOTAL.labels(status=status).inc()
        if validation.risk_violations:
            for violation in validation.risk_violations:
                SUITABILITY_VIOLATIONS_TOTAL.labels(rule=violation).inc()
        
        # 4. Update overall stats
        performance_stats['total_requests'] += 1
        if status == 'APPROVED':
            performance_stats['total_approved'] += 1
        else:
            performance_stats['total_rejected'] += 1
            
        current_avg = performance_stats['avg_response_time']
        n = performance_stats['total_requests']
        performance_stats['avg_response_time'] = round((current_avg * (n - 1) + (end_time - start_time)) / n, 4)
        
        response_payload = {
            "request_id": f"req_{datetime.utcnow().timestamp():.0f}_{request.investor_id}",
            "investor_id": request.investor_id,
            "investor_summary": {
                "age": investor_profile.age,
                "risk_tolerance": investor_profile.risk_tolerance,
                "experience": investor_profile.investment_experience,
                "time_horizon": investor_profile.time_horizon_years
            },
            "status": status,
            "recommendation": None if status == "REJECTED" else {
                "products": validation.approved_products,
                "allocations": [
                    recommendation.allocation_percentages[
                        recommendation.recommended_products.index(ticker)
                    ]
                    for ticker in validation.approved_products
                ],
                "rationale": recommendation.rationale
            },
            "validation_summary": {
                "is_suitable": validation.is_suitable,
                "portfolio_risk": validation.overall_portfolio_risk,
                "violations": validation.risk_violations,
                "compliance_notes": validation.compliance_notes
            },
            "timestamp": datetime.utcnow().isoformat(),
            "performance_metrics": performance_metrics,
            "data_source": data_source
        }
        
        # 5. Background Task: compliance audit logging
        background_tasks.add_task(
            log_audit,
            request_id=response_payload['request_id'],
            profile=investor_profile,
            query=request.query,
            recommendation=recommendation.dict() if recommendation else {},
            validation=validation.dict(),
            status=status,
            metrics=performance_metrics
        )
        
        return response_payload
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consultation process failure: {str(e)}")

# ============================================================================
# CLIENT & PRODUCT BROWSER ENDPOINTS
# ============================================================================

@app.get("/investors")
async def list_investors(limit: int = 100):
    """Retrieve lists of investor client profiles"""
    # Try fetching from Database
    investors = []
    data_source = "json_file"
    
    if db_manager and db_manager.pool:
        investors = db_manager.get_all_investors(limit=limit)
        data_source = "postgresql"
        
    # Fallback to local cache
    if not investors:
        investors = list(investor_profiles_fallback.values())[:limit]
        
    return {
        "total_count": len(investors),
        "data_source": data_source,
        "investors": [
            {
                "investor_id": inv.investor_id,
                "age": inv.age,
                "risk_tolerance": inv.risk_tolerance,
                "experience": inv.investment_experience,
                "time_horizon": inv.time_horizon_years
            }
            for inv in investors
        ]
    }

@app.get("/investors/{investor_id}")
async def get_investor(investor_id: str):
    """Retrieve detailed statistics for single investor client"""
    investor = None
    data_source = "json_file"
    
    if db_manager and db_manager.pool:
        investor = db_manager.get_investor_profile(investor_id)
        data_source = "postgresql"
        
    if not investor:
        investor = investor_profiles_fallback.get(investor_id)
        
    if not investor:
        raise HTTPException(status_code=404, detail="Investor profile not found")
        
    suitable_products = validator.get_suitable_products(investor) if validator else []
    
    return {
        "investor_id": investor.investor_id,
        "profile": {
            "age": investor.age,
            "risk_tolerance": investor.risk_tolerance,
            "annual_income": investor.annual_income,
            "net_worth": investor.net_worth,
            "investment_experience": investor.investment_experience,
            "time_horizon_years": investor.time_horizon_years
        },
        "suitable_products_count": len(suitable_products),
        "suitable_products": suitable_products,
        "data_source": data_source
    }

@app.get("/products")
async def list_products():
    """Retrieve catalog products"""
    products = []
    data_source = "json_file"
    
    if db_manager and db_manager.pool:
        products = db_manager.get_all_products()
        data_source = "postgresql"
        
    if not products:
        products = products_fallback
        
    return {
        "total_count": len(products),
        "data_source": data_source,
        "products": [
            {
                "ticker": p.ticker,
                "name": p.name,
                "risk_score": p.risk_score,
                "asset_class": p.asset_class,
                "min_investment": p.min_investment
            }
            for p in products
        ]
    }

# ============================================================================
# AUDIT & COMPLIANCE VERIFICATION ENDPOINTS
# ============================================================================

@app.post("/audit/simulate")
async def run_fairness_audit():
    """Triggers an on-the-fly fairness simulation calculation auditing demographic parity"""
    if not validator:
        raise HTTPException(status_code=503, detail="Validation engine uninitialized.")
        
    # Get all active investors
    investors = []
    if db_manager and db_manager.pool:
        investors = db_manager.get_all_investors(limit=100)
    if not investors:
        investors = list(investor_profiles_fallback.values())
        
    # Generate 8 test portfolio recommendations (diverse risk spectrums)
    test_recommendations = generate_diverse_test_recommendations()
    
    # Execute fairness audit computations
    auditor = FairnessAuditor(validator)
    simulation_results = auditor.run_comprehensive_simulation(investors, test_recommendations)
    metrics = auditor.calculate_comprehensive_metrics(simulation_results)
    
    return metrics

@app.get("/audit/stats")
async def get_audit_stats():
    """Retrieve compliance stats from log"""
    total = len(audit_log_fallback)
    approved = sum(1 for log in audit_log_fallback if log['status'] == 'APPROVED')
    rejected = sum(1 for log in audit_log_fallback if log['status'] == 'REJECTED')
    
    # Try fetching aggregate statistics from MongoDB
    if mongo_logger and mongo_logger.is_available():
        try:
            mongo_stats = mongo_logger.get_statistics()
            return mongo_stats
        except:
            pass
            
    return {
        "total_requests": total,
        "approved_count": approved,
        "rejected_count": rejected,
        "approval_rate": approved / total if total > 0 else 0.0,
        "rejection_rate": rejected / total if total > 0 else 0.0
    }

# ============================================================================
# COMPLIANCE LOGGING HELPERS
# ============================================================================

def log_audit(request_id: str, profile: InvestorProfile, query: str, recommendation: Dict, validation: Dict, status: str, metrics: Dict):
    """Compliance background logging utility (MongoDB audit trace with local JSON fallback)"""
    global audit_log_fallback
    
    log_entry = {
        "request_id": request_id,
        "investor_id": profile.investor_id,
        "investor_age": profile.age,
        "investor_risk": profile.risk_tolerance,
        "query": query,
        "recommendation": recommendation,
        "validation_result": validation,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "performance_metrics": metrics
    }
    
    # Tier 1: Log to MongoDB
    logged_mongodb = False
    if mongo_logger and mongo_logger.is_available():
        try:
            logged_mongodb = mongo_logger.log_recommendation(
                request_id=request_id,
                investor_profile=profile,
                query=query,
                recommendation=recommendation,
                validation_result=validation,
                status=status,
                performance_metrics=metrics
            )
        except Exception as e:
            print(f"⚠️  MongoDB logging failed: {e}")
            
    # Tier 2: Local memory cache
    audit_log_fallback.append(log_entry)
    
    # Periodically flush fallback log to disk
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/audit_logs.json', 'w') as f:
            json.dump(audit_log_fallback[-1000:], f, indent=2) # keep last 1000 logs on disk
    except Exception as e:
        print(f"⚠️  Failed to save local audit_logs.json: {e}")

# ============================================================================
# RUN SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    # Dynamically bind port from core config
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)
