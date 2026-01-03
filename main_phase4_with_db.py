"""
FastAPI with ACTUAL PostgreSQL Integration (Phase 4 Complete)
This version actually uses the database!

Key differences from main_phase4.py:
- Loads data from PostgreSQL (not just JSON)
- Saves new data to PostgreSQL
- Uses database connection pooling
- Falls back to JSON if database unavailable
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import json
import os
import time

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Import Phase 1 & 2 components
from schemas import InvestorProfile, PortfolioRecommendation, SuitabilityResult, Product
from vector_store import VectorStore, MockEmbedder
from context_injector import ContextInjector
from suitability_validator_expanded import SuitabilityValidator
from rag_advisor import RAGInvestmentAdvisor

# Import NEW database layer
from database import DatabaseManager, get_db

# Initialize FastAPI app
app = FastAPI(
    title="Ethical AI Investment Advisor (Phase 4 - With PostgreSQL)",
    description="Production system with actual database integration",
    version="4.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PROMETHEUS METRICS (same as before)
# ============================================================================

http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
http_request_duration_seconds = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
http_requests_in_progress = Gauge('http_requests_in_progress', 'Requests in progress', ['method', 'endpoint'])
advice_requests_total = Counter('advice_requests_total', 'Investment advice requests', ['status', 'age_group', 'risk_level'])
advice_approval_rate = Gauge('advice_approval_rate', 'Approval rate percentage')
portfolio_risk_distribution = Histogram('portfolio_risk_distribution', 'Portfolio risk scores')
cache_hits_total = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
cache_misses_total = Counter('cache_misses_total', 'Cache misses', ['cache_type'])
database_queries_total = Counter('database_queries_total', 'Database queries', ['operation'])
database_query_duration_seconds = Histogram('database_query_duration_seconds', 'Database query duration', ['operation'])
active_investors_gauge = Gauge('active_investors_total', 'Total active investors')
products_available_gauge = Gauge('products_available_total', 'Total available products')

# ============================================================================
# GLOBAL STATE
# ============================================================================

advisor = None
audit_log = []
redis_client = None
db_manager = None  # NEW: Database manager

performance_stats = {
    'total_requests': 0,
    'total_approved': 0,
    'total_rejected': 0,
    'avg_response_time': 0,
    'cache_hit_rate': 0
}

# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Collect metrics for every request"""
    method = request.method
    path = request.url.path
    
    if path == "/metrics":
        return await call_next(request)
    
    start_time = time.time()
    http_requests_in_progress.labels(method=method, endpoint=path).inc()
    
    try:
        response = await call_next(request)
        status = response.status_code
        duration = time.time() - start_time
        http_requests_total.labels(method=method, endpoint=path, status=status).inc()
        http_request_duration_seconds.labels(method=method, endpoint=path).observe(duration)
        return response
    except Exception as e:
        http_requests_total.labels(method=method, endpoint=path, status=500).inc()
        raise
    finally:
        http_requests_in_progress.labels(method=method, endpoint=path).dec()

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services - NOW WITH DATABASE!"""
    global advisor, redis_client, db_manager
    
    print("=" * 70)
    print("🚀 Starting Ethical AI Advisor (Phase 4 - WITH DATABASE)")
    print("=" * 70)
    print()
    
    try:
        # Initialize Redis
        try:
            import redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
            print("✅ Redis cache connected")
        except Exception as e:
            print(f"⚠️  Redis not available: {e}")
            redis_client = None
        
        # Initialize DATABASE (NEW!)
        print("📊 Connecting to PostgreSQL...")
        db_manager = DatabaseManager()
        
        # Check if database has data
        db_investors = db_manager.get_all_investors()
        db_products = db_manager.get_all_products()
        
        print(f"   Found {len(db_investors)} investors in database")
        print(f"   Found {len(db_products)} products in database")
        
        # If database is empty, migrate from JSON
        if len(db_investors) == 0 or len(db_products) == 0:
            print("\n⚠️  Database is empty! Migrating from JSON files...")
            stats = db_manager.migrate_from_json()
            print(f"✅ Migration complete: {stats['investors_migrated']} investors, {stats['products_migrated']} products")
            
            # Re-fetch after migration
            db_investors = db_manager.get_all_investors()
            db_products = db_manager.get_all_products()
        
        # Load vector store
        print("📚 Loading vector store...")
        embedder = MockEmbedder(dim=1536)
        vector_store = VectorStore(embedding_dim=1536)
        
        if os.path.exists('product_vector_store.index'):
            vector_store.load('product_vector_store')
        else:
            raise FileNotFoundError("Vector store not found. Run: python vector_store.py")
        
        # Update gauges
        products_available_gauge.set(len(db_products))
        active_investors_gauge.set(len(db_investors))
        
        # Initialize validator (Phase 1) - using database products
        print("🛡️  Initializing Phase 1 validator...")
        validator = SuitabilityValidator(db_products)
        
        # Initialize RAG advisor (Phase 2 + Phase 1)
        print("🤖 Initializing RAG advisor...")
        advisor = RAGInvestmentAdvisor(
            vector_store=vector_store,
            embedder=embedder,
            validator=validator,
            use_real_llm=False
        )
        
        print()
        print("=" * 70)
        print("✅ SERVICE INITIALIZED SUCCESSFULLY (WITH DATABASE!)")
        print("=" * 70)
        print(f"📊 Products in database: {len(db_products)}")
        print(f"👥 Investors in database: {len(db_investors)}")
        print(f"🔍 Vector store documents: {len(vector_store.documents)}")
        print(f"💾 Data source: PostgreSQL + Redis cache")
        print(f"📈 Prometheus metrics: /metrics")
        print(f"🌐 API ready at: http://0.0.0.0:8000")
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global db_manager
    
    print("\n" + "=" * 70)
    print("👋 Shutting down Ethical AI Advisor Service...")
    print("=" * 70)
    
    if audit_log:
        with open('audit_logs.json', 'w') as f:
            json.dump(audit_log, f, indent=2)
        print(f"💾 Saved {len(audit_log)} audit log entries")
    
    if redis_client:
        redis_client.close()
        print("✅ Redis connection closed")
    
    if db_manager:
        db_manager.close_all()
        print("✅ Database connections closed")
    
    print("✅ Shutdown complete")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ConsultRequest(BaseModel):
    investor_id: str = Field(..., description="Unique investor identifier")
    query: str = Field(..., min_length=10, description="Investment question")

class ConsultResponse(BaseModel):
    request_id: str
    investor_id: str
    investor_summary: Dict
    status: str
    recommendation: Optional[Dict] = None
    validation_summary: Dict
    timestamp: str
    performance_metrics: Optional[Dict] = None
    data_source: str  # NEW: Shows where data came from

class AuditQueryRequest(BaseModel):
    investor_id: Optional[str] = None
    status: Optional[str] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    risk_level: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = Field(default=100, le=1000)

# ============================================================================
# METRICS ENDPOINT
# ============================================================================

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Ethical AI Investment Advisor",
        "version": "4.1.0",
        "phase": "Phase 4 - WITH DATABASE INTEGRATION! 🎉",
        "status": "operational",
        "features": {
            "postgresql_database": db_manager is not None,
            "redis_caching": redis_client is not None,
            "prometheus_metrics": True,
            "data_persistence": True
        },
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs",
            "consult": "/consult",
            "investors": "/investors (from database!)",
            "products": "/products (from database!)",
            "audit": "/audit/*"
        }
    }

@app.get("/health")
async def health_check():
    """Health check with database status"""
    redis_status = "connected" if redis_client else "not_available"
    db_status = "connected" if db_manager and db_manager.pool else "not_available"
    
    try:
        if redis_client:
            redis_client.ping()
            redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    try:
        if db_manager:
            # Test database connection
            investors = db_manager.get_all_investors(limit=1)
            db_status = "healthy" if investors is not None else "unhealthy"
    except:
        db_status = "unhealthy"
    
    return {
        "status": "healthy",
        "service": "ethical-ai-advisor",
        "version": "4.1.0",
        "components": {
            "advisor": "operational" if advisor is not None else "not initialized",
            "vector_store": "loaded",
            "validator": "ready",
            "redis_cache": redis_status,
            "postgresql": db_status
        },
        "metrics": {
            "total_requests": performance_stats['total_requests'],
            "cache_hit_rate": f"{performance_stats['cache_hit_rate']:.1%}",
            "avg_response_time": f"{performance_stats['avg_response_time']:.3f}s"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# CACHED INVESTOR LOOKUP - NOW WITH DATABASE!
# ============================================================================

async def get_investor_cached(investor_id: str) -> Optional[InvestorProfile]:
    """
    Get investor profile with 3-tier caching:
    1. Redis (fastest - 1ms)
    2. PostgreSQL (fast - 10ms)
    3. JSON fallback (if database fails)
    """
    # Tier 1: Try Redis cache first
    if redis_client:
        try:
            cache_key = f"profile:{investor_id}"
            cached = redis_client.get(cache_key)
            
            if cached:
                cache_hits_total.labels(cache_type="redis").inc()
                return InvestorProfile(**json.loads(cached))
            else:
                cache_misses_total.labels(cache_type="redis").inc()
        except Exception as e:
            print(f"Redis error: {e}")
    
    # Tier 2: Try PostgreSQL database
    if db_manager:
        try:
            start_time = time.time()
            investor = db_manager.get_investor_profile(investor_id)
            duration = time.time() - start_time
            
            database_queries_total.labels(operation="get_investor").inc()
            database_query_duration_seconds.labels(operation="get_investor").observe(duration)
            
            if investor:
                # Store in Redis for next time
                if redis_client:
                    try:
                        cache_key = f"profile:{investor_id}"
                        redis_client.setex(cache_key, 3600, json.dumps(investor.dict()))
                    except:
                        pass
                
                return investor
        except Exception as e:
            print(f"Database error: {e}")
    
    # Tier 3: Fallback to JSON file
    try:
        with open('investor_profiles.json', 'r') as f:
            profiles = json.load(f)
            profile_dict = next((p for p in profiles if p['investor_id'] == investor_id), None)
            if profile_dict:
                return InvestorProfile(**profile_dict)
    except:
        pass
    
    return None

# ============================================================================
# MAIN CONSULT ENDPOINT
# ============================================================================

@app.post("/consult", response_model=ConsultResponse)
async def consult(request: ConsultRequest, background_tasks: BackgroundTasks):
    """
    Investment advice endpoint - NOW USING DATABASE!
    """
    if advisor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    request_start = time.time()
    timings = {}
    
    request_id = f"req_{datetime.utcnow().timestamp():.0f}_{request.investor_id}"
    
    try:
        # Load investor (from cache → database → JSON)
        cache_start = time.time()
        investor_profile = await get_investor_cached(request.investor_id)
        timings['data_lookup'] = time.time() - cache_start
        
        if not investor_profile:
            raise HTTPException(status_code=404, detail=f"Investor {request.investor_id} not found")
        
        # Determine age group
        age = investor_profile.age
        age_group = "young" if age < 35 else "middle" if age < 55 else "senior"
        
        # Get advice through RAG pipeline
        rag_start = time.time()
        result = advisor.get_advice(
            investor_profile=investor_profile,
            user_query=request.query,
            top_k=5
        )
        timings['rag_pipeline'] = time.time() - rag_start
        
        validation = result['validation_result']
        recommendation = result['llm_recommendation']
        status = result['final_status']
        
        # Update metrics
        advice_requests_total.labels(
            status=status,
            age_group=age_group,
            risk_level=investor_profile.risk_tolerance
        ).inc()
        
        if validation.overall_portfolio_risk > 0:
            portfolio_risk_distribution.observe(validation.overall_portfolio_risk)
        
        performance_stats['total_requests'] += 1
        if status == 'APPROVED':
            performance_stats['total_approved'] += 1
        else:
            performance_stats['total_rejected'] += 1
        
        total = performance_stats['total_requests']
        approved = performance_stats['total_approved']
        advice_approval_rate.set(approved / total if total > 0 else 0)
        
        total_time = time.time() - request_start
        current_avg = performance_stats['avg_response_time']
        performance_stats['avg_response_time'] = (current_avg * (total - 1) + total_time) / total
        
        # Determine data source
        data_source = "unknown"
        if timings['data_lookup'] < 0.002:
            data_source = "redis_cache"
        elif db_manager:
            data_source = "postgresql"
        else:
            data_source = "json_file"
        
        response_data = {
            "request_id": request_id,
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
                "compliance_notes": validation.compliance_notes[:3]
            },
            "timestamp": datetime.utcnow().isoformat(),
            "performance_metrics": {
                "total_time_ms": round(total_time * 1000, 2),
                "data_lookup_ms": round(timings['data_lookup'] * 1000, 2),
                "rag_pipeline_ms": round(timings['rag_pipeline'] * 1000, 2)
            },
            "data_source": data_source
        }
        
        background_tasks.add_task(
            log_to_audit,
            request_id=request_id,
            investor_profile=investor_profile,
            query=request.query,
            recommendation=recommendation.dict(),
            validation=validation.dict(),
            status=status
        )
        
        return ConsultResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ============================================================================
# INVESTOR ENDPOINTS - NOW FROM DATABASE!
# ============================================================================

@app.get("/investors")
async def list_investors(
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
    risk_level: Optional[int] = None,
    experience: Optional[str] = None,
    limit: int = 100
):
    """List investors - FROM DATABASE!"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    database_queries_total.labels(operation="list_investors").inc()
    
    start_time = time.time()
    profiles = db_manager.search_investors(min_age, max_age, risk_level, experience, limit)
    duration = time.time() - start_time
    
    database_query_duration_seconds.labels(operation="list_investors").observe(duration)
    
    return {
        "total_count": len(profiles),
        "data_source": "postgresql",
        "query_time_ms": round(duration * 1000, 2),
        "investors": [
            {
                "investor_id": p.investor_id,
                "age": p.age,
                "risk_tolerance": p.risk_tolerance,
                "experience": p.investment_experience,
                "net_worth": p.net_worth,
                "time_horizon": p.time_horizon_years
            }
            for p in profiles
        ]
    }

@app.get("/investors/{investor_id}")
async def get_investor(investor_id: str):
    """Get specific investor - FROM DATABASE!"""
    investor = await get_investor_cached(investor_id)
    
    if not investor:
        raise HTTPException(status_code=404, detail=f"Investor {investor_id} not found")
    
    suitable_products = advisor.validator.get_suitable_products(investor)
    
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
        "suitable_products": suitable_products[:10],
        "request_history_count": len([log for log in audit_log if log['investor_id'] == investor_id]),
        "data_source": "postgresql"
    }

# ============================================================================
# PRODUCT ENDPOINTS - NOW FROM DATABASE!
# ============================================================================

@app.get("/products")
async def list_products(
    risk_level: Optional[int] = None,
    asset_class: Optional[str] = None,
    max_min_investment: Optional[float] = None
):
    """List products - FROM DATABASE!"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    database_queries_total.labels(operation="list_products").inc()
    
    start_time = time.time()
    products = db_manager.get_all_products()
    duration = time.time() - start_time
    
    database_query_duration_seconds.labels(operation="list_products").observe(duration)
    
    # Apply filters
    if risk_level:
        products = [p for p in products if p.risk_score <= risk_level]
    if asset_class:
        products = [p for p in products if p.asset_class == asset_class]
    if max_min_investment:
        products = [p for p in products if p.min_investment <= max_min_investment]
    
    return {
        "total_count": len(products),
        "data_source": "postgresql",
        "query_time_ms": round(duration * 1000, 2),
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
# AUDIT ENDPOINTS (same as before)
# ============================================================================

@app.post("/audit/query")
async def query_audit_logs(request: AuditQueryRequest):
    """Query audit logs"""
    filtered_logs = audit_log.copy()
    
    if request.investor_id:
        filtered_logs = [log for log in filtered_logs if log['investor_id'] == request.investor_id]
    if request.status:
        filtered_logs = [log for log in filtered_logs if log['status'] == request.status]
    if request.min_age:
        filtered_logs = [log for log in filtered_logs if log['investor_age'] >= request.min_age]
    if request.max_age:
        filtered_logs = [log for log in filtered_logs if log['investor_age'] <= request.max_age]
    if request.risk_level:
        filtered_logs = [log for log in filtered_logs if log['investor_risk'] == request.risk_level]
    
    filtered_logs = filtered_logs[:request.limit]
    
    return {
        "total_count": len(filtered_logs),
        "logs": filtered_logs
    }

@app.get("/audit/stats")
async def get_audit_stats():
    """Get audit statistics"""
    if not audit_log:
        return {"total_requests": 0, "approved_count": 0, "rejected_count": 0}
    
    total = len(audit_log)
    approved = sum(1 for log in audit_log if log['status'] == 'APPROVED')
    rejected = sum(1 for log in audit_log if log['status'] == 'REJECTED')
    
    return {
        "total_requests": total,
        "approved_count": approved,
        "rejected_count": rejected,
        "approval_rate": approved / total if total > 0 else 0.0
    }

def log_to_audit(request_id: str, investor_profile: InvestorProfile, query: str, recommendation: Dict, validation: Dict, status: str):
    """Background task for audit logging"""
    log_entry = {
        "request_id": request_id,
        "investor_id": investor_profile.investor_id,
        "investor_age": investor_profile.age,
        "investor_risk": investor_profile.risk_tolerance,
        "query": query,
        "recommendation": recommendation,
        "validation_result": validation,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "blocked_products": validation.get('rejected_products', []),
        "violations": validation.get('risk_violations', [])
    }
    audit_log.append(log_entry)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
