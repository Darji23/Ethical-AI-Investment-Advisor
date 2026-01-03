"""
FastAPI Application for Ethical AI Investment Advisor (Phase 3)
Integrates Phase 1 (Validation) + Phase 2 (RAG with Jina & Fireworks)

Provides REST API endpoints for investment advice with full safety validation
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import json
import os

# Phase 1 & 2 imports
from schemas import InvestorProfile, PortfolioRecommendation, SuitabilityResult, Product
from vector_store import VectorStore
from suitability_validator_expanded import SuitabilityValidator
from rag_advisor import RAGInvestmentAdvisor
from jina_embedder import JinaEmbedder
from fireworks_llm import FireworksLLM

# Initialize FastAPI
app = FastAPI(
    title="Ethical AI Investment Advisor",
    description="Production API with Jina AI embeddings + Fireworks AI LLM",
    version="3.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
advisor = None
investor_profiles_cache = {}
products = []
audit_log = []

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ConsultRequest(BaseModel):
    investor_id: str = Field(..., description="Unique investor identifier")
    query: str = Field(..., min_length=10, description="Investment question or goal")

class ConsultResponse(BaseModel):
    request_id: str
    investor_id: str
    investor_summary: Dict
    status: str  # APPROVED or REJECTED
    recommendation: Optional[Dict] = None
    validation_summary: Dict
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    components: Dict
    timestamp: str

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize all services"""
    global advisor, investor_profiles_cache, products
    
    print("=" * 70)
    print("🚀 Starting Ethical AI Advisor API (Phase 3)")
    print("=" * 70)
    print()
    
    try:
        # Load investor profiles
        print("📊 Loading investor profiles...")
        with open('investor_profiles.json', 'r') as f:
            profiles_data = json.load(f)
            for profile_dict in profiles_data:
                investor_profiles_cache[profile_dict['investor_id']] = InvestorProfile(**profile_dict)
        
        print(f"✅ Loaded {len(investor_profiles_cache)} investor profiles")
        
        # Load products
        print("💼 Loading products...")
        with open('product_master.json', 'r') as f:
            products_data = json.load(f)
            products = [Product(**p) for p in products_data]
        
        print(f"✅ Loaded {len(products)} products")
        
        # Initialize Jina embedder
        print("🤖 Initializing Jina AI embedder...")
        embedder = JinaEmbedder()
        print("✅ Jina AI embedder ready")
        
        # Initialize Fireworks LLM
        print("🤖 Initializing Fireworks AI LLM...")
        llm = FireworksLLM()
        print("✅ Fireworks AI LLM ready")
        
        # Load vector store
        print("📚 Loading vector store...")
        vector_store = VectorStore(embedding_dim=1024)
        
        if os.path.exists('product_vector_store.index'):
            vector_store.load('product_vector_store')
            print(f"✅ Vector store loaded ({len(vector_store.documents)} products)")
        else:
            raise FileNotFoundError("Vector store not found. Run: python vector_store.py")
        
        # Initialize validator (Phase 1)
        print("🛡️  Initializing Phase 1 validator...")
        validator = SuitabilityValidator(products)
        print("✅ Validator ready")
        
        # Initialize RAG advisor (Phase 2)
        print("🎯 Initializing RAG advisor...")
        advisor = RAGInvestmentAdvisor(
            vector_store=vector_store,
            embedder=embedder,
            llm=llm,
            validator=validator
        )
        print("✅ RAG advisor ready")
        
        print()
        print("=" * 70)
        print("✅ API INITIALIZED SUCCESSFULLY")
        print("=" * 70)
        print(f"📊 Investors: {len(investor_profiles_cache)}")
        print(f"💼 Products: {len(products)}")
        print(f"🔍 Vector store: {len(vector_store.documents)} products indexed")
        print(f"🤖 AI: Jina AI + Fireworks AI (Llama 3.1 70B)")
        print(f"🌐 API ready at: http://0.0.0.0:8000")
        print(f"📖 Docs at: http://0.0.0.0:8000/docs")
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n" + "=" * 70)
    print("👋 Shutting down API service...")
    print("=" * 70)
    
    # Save audit logs
    if audit_log:
        with open('audit_logs.json', 'w') as f:
            json.dump(audit_log, f, indent=2)
        print(f"💾 Saved {len(audit_log)} audit log entries")
    
    print("✅ Shutdown complete")

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Ethical AI Investment Advisor",
        "version": "3.0.0",
        "phase": "Phase 3 - Microservices with Real AI",
        "status": "operational",
        "ai_models": {
            "embeddings": "Jina AI (1024-dim)",
            "llm": "Fireworks AI (Llama 3.1 70B)"
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "consult": "/consult",
            "investors": "/investors",
            "products": "/products",
            "audit": "/audit/*"
        },
        "message": "Providing ethical AI-powered investment advice with dual-layer validation"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="ethical-ai-advisor",
        version="3.0.0",
        components={
            "advisor": "operational" if advisor is not None else "not initialized",
            "vector_store": "loaded",
            "validator": "ready",
            "jina_ai": "connected",
            "fireworks_ai": "connected"
        },
        timestamp=datetime.utcnow().isoformat()
    )

# ============================================================================
# MAIN CONSULTATION ENDPOINT
# ============================================================================

@app.post("/consult", response_model=ConsultResponse)
async def consult(request: ConsultRequest, background_tasks: BackgroundTasks):
    """
    Get investment advice with full validation
    
    Flow:
    1. Load investor profile
    2. RAG pipeline (Jina + Fireworks + Validator)
    3. Return recommendation with validation results
    """
    if advisor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Generate request ID
    request_id = f"req_{datetime.utcnow().timestamp():.0f}_{request.investor_id}"
    
    try:
        # Get investor profile
        investor_profile = investor_profiles_cache.get(request.investor_id)
        
        if not investor_profile:
            raise HTTPException(
                status_code=404,
                detail=f"Investor {request.investor_id} not found"
            )
        
        # Get advice through RAG pipeline
        result = advisor.get_advice(
            investor_profile=investor_profile,
            user_query=request.query,
            top_k=5
        )
        
        # Extract results
        validation = result['validation_result']
        recommendation = result['llm_recommendation']
        status = result['final_status']
        
        # Build response
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
                "compliance_notes": validation.compliance_notes[:3]  # First 3 notes
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log to audit (background task)
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
# INVESTOR ENDPOINTS
# ============================================================================

@app.get("/investors")
async def list_investors(
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
    risk_level: Optional[int] = None,
    experience: Optional[str] = None,
    limit: int = 100
):
    """List investors with optional filters"""
    filtered = list(investor_profiles_cache.values())
    
    # Apply filters
    if min_age:
        filtered = [inv for inv in filtered if inv.age >= min_age]
    if max_age:
        filtered = [inv for inv in filtered if inv.age <= max_age]
    if risk_level:
        filtered = [inv for inv in filtered if inv.risk_tolerance == risk_level]
    if experience:
        filtered = [inv for inv in filtered if inv.investment_experience == experience]
    
    # Limit results
    filtered = filtered[:limit]
    
    return {
        "total_count": len(filtered),
        "investors": [
            {
                "investor_id": inv.investor_id,
                "age": inv.age,
                "risk_tolerance": inv.risk_tolerance,
                "experience": inv.investment_experience,
                "net_worth": inv.net_worth
            }
            for inv in filtered
        ]
    }

@app.get("/investors/{investor_id}")
async def get_investor(investor_id: str):
    """Get specific investor details"""
    investor = investor_profiles_cache.get(investor_id)
    
    if not investor:
        raise HTTPException(status_code=404, detail=f"Investor {investor_id} not found")
    
    # Get suitable products
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
        "suitable_products": suitable_products[:10],  # First 10
        "request_history_count": len([log for log in audit_log if log['investor_id'] == investor_id])
    }

# ============================================================================
# PRODUCT ENDPOINTS
# ============================================================================

@app.get("/products")
async def list_products(
    risk_level: Optional[int] = None,
    asset_class: Optional[str] = None,
    max_min_investment: Optional[float] = None
):
    """List products with optional filters"""
    filtered = products.copy()
    
    # Apply filters
    if risk_level:
        filtered = [p for p in filtered if p.risk_score <= risk_level]
    if asset_class:
        filtered = [p for p in filtered if p.asset_class == asset_class]
    if max_min_investment:
        filtered = [p for p in filtered if p.min_investment <= max_min_investment]
    
    return {
        "total_count": len(filtered),
        "products": [
            {
                "ticker": p.ticker,
                "name": p.name,
                "risk_score": p.risk_score,
                "asset_class": p.asset_class,
                "min_investment": p.min_investment
            }
            for p in filtered
        ]
    }

@app.get("/products/{ticker}")
async def get_product(ticker: str):
    """Get specific product details"""
    product = next((p for p in products if p.ticker == ticker), None)
    
    if not product:
        raise HTTPException(status_code=404, detail=f"Product {ticker} not found")
    
    return {
        "ticker": product.ticker,
        "name": product.name,
        "risk_score": product.risk_score,
        "asset_class": product.asset_class,
        "min_investment": product.min_investment
    }

# ============================================================================
# AUDIT ENDPOINTS
# ============================================================================

@app.get("/audit/logs")
async def get_audit_logs(
    investor_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Get audit logs with optional filters"""
    filtered = audit_log.copy()
    
    if investor_id:
        filtered = [log for log in filtered if log['investor_id'] == investor_id]
    if status:
        filtered = [log for log in filtered if log['status'] == status]
    
    filtered = filtered[:limit]
    
    return {
        "total_count": len(filtered),
        "logs": filtered
    }

@app.get("/audit/stats")
async def get_audit_stats():
    """Get audit statistics"""
    if not audit_log:
        return {
            "total_requests": 0,
            "approved_count": 0,
            "rejected_count": 0,
            "approval_rate": 0.0
        }
    
    total = len(audit_log)
    approved = sum(1 for log in audit_log if log['status'] == 'APPROVED')
    rejected = sum(1 for log in audit_log if log['status'] == 'REJECTED')
    
    return {
        "total_requests": total,
        "approved_count": approved,
        "rejected_count": rejected,
        "approval_rate": approved / total if total > 0 else 0.0,
        "rejection_rate": rejected / total if total > 0 else 0.0
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_to_audit(
    request_id: str,
    investor_profile: InvestorProfile,
    query: str,
    recommendation: Dict,
    validation: Dict,
    status: str
):
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
    print(f"📝 Logged audit entry: {request_id} - {status}")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
