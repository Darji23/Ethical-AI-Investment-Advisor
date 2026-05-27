"""
RAG Investment Advisor
Complete pipeline: Jina Embeddings → FAISS → Fireworks LLM → Validator
Includes automatic fallback to mock services if API keys are missing.
Resolves parameter swapping bugs in post-validation.
"""
from typing import Dict, List
import numpy as np

from app.schemas.schemas import InvestorProfile, PortfolioRecommendation, SuitabilityResult, Product
from app.services.vector_store import VectorStore, ProductDocument
from app.services.context_injector import ContextInjector
from app.services.suitability_validator import SuitabilityValidator

# Import real AI models & get_embedder/get_llm helpers
from app.services.jina_embedder import JinaEmbedder, MockEmbedder, get_embedder
from app.services.fireworks_llm import FireworksLLM, MockLLM, get_llm

class RAGInvestmentAdvisor:
    """
    Complete RAG Pipeline with Robust Fallbacks and Regulation Validation
    
    Flow:
    1. Embed user query (Jina AI / Mock)
    2. Retrieve relevant products (FAISS)
    3. Filter by risk/age constraints (Validator)
    4. Generate recommendation (Fireworks AI / Mock)
    5. Final validation (Validator - Swapped parameters bug resolved)
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder=None,
        llm=None,
        validator: SuitabilityValidator = None,
        use_real_ai: bool = True
    ):
        """
        Initialize RAG advisor
        """
        self.vector_store = vector_store
        self.validator = validator
        self.context_injector = ContextInjector()
        
        # Resolve embedder with fallback
        if embedder is None:
            self.embedder = get_embedder(use_real=use_real_ai)
        else:
            self.embedder = embedder
        
        # Resolve LLM with fallback
        if llm is None:
            self.llm = get_llm(use_real=use_real_ai)
        else:
            self.llm = llm
        
        print(f"\n{'='*70}")
        print("🤖 RAG Investment Advisor Initialized")
        print(f"{'='*70}")
        print(f"Embedder: {type(self.embedder).__name__}")
        print(f"LLM: {type(self.llm).__name__}")
        print(f"Vector Store: {len(self.vector_store.documents)} products indexed")
        print(f"Validator: {'Enabled' if self.validator else 'Disabled'}")
        print(f"{'='*70}\n")
    
    def get_advice(
        self,
        investor_profile: InvestorProfile,
        user_query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Generate investment advice using complete RAG pipeline
        """
        print(f"\n{'─'*70}")
        print(f"🎯 Processing request for {investor_profile.investor_id}")
        print(f"   Query: {user_query}")
        print(f"{'─'*70}\n")
        
        # STEP 1: PRE-FILTERING (Tier 1 Safety)
        print("🛡️  STEP 1: Pre-retrieval risk filtering...")
        max_risk = self.validator.get_max_risk_for_investor(investor_profile) if self.validator else 5
        
        eligible_products = [
            doc for doc in self.vector_store.documents
            if doc.risk_score <= max_risk
        ]
        
        print(f"   Max risk allowed: {max_risk}/5")
        print(f"   Eligible products: {len(eligible_products)}/{len(self.vector_store.documents)}")
        
        if not eligible_products:
            return self._create_rejection_response(
                investor_profile,
                "No products match investor's risk profile",
                []
            )
        
        # STEP 2: SEMANTIC SEARCH (Jina AI)
        print("\n🔍 STEP 2: Semantic search with Jina AI...")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(user_query)
        
        # Search in eligible products
        eligible_indices = [
            self.vector_store.documents.index(doc)
            for doc in eligible_products
        ]
        
        # Get embeddings for eligible products
        eligible_embeddings = np.array([
            self.vector_store.index.reconstruct(i)
            for i in eligible_indices
        ])
        
        # Calculate similarities (cosine similarity)
        from numpy.linalg import norm
        similarities = np.array([
            np.dot(query_embedding, emb) / (norm(query_embedding) * norm(emb)) if norm(query_embedding) > 0 and norm(emb) > 0 else 0
            for emb in eligible_embeddings
        ])
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_docs = [eligible_products[i] for i in top_indices]
        
        print(f"   Retrieved {len(retrieved_docs)} relevant products:")
        for i, doc in enumerate(retrieved_docs, 1):
            sim = similarities[top_indices[i-1]]
            print(f"   {i}. {doc.ticker} - {doc.name[:40]:40} (similarity: {sim:.3f})")
        
        # STEP 3: GENERATE RECOMMENDATION (Fireworks AI)
        print("\n🤖 STEP 3: Generating recommendation with Fireworks AI...")
        
        # Get constraints
        constraints = self.context_injector.get_constraints(investor_profile)
        
        # Convert docs to dict format
        products_for_llm = [
            {
                "ticker": doc.ticker,
                "name": doc.name,
                "risk_score": doc.risk_score,
                "asset_class": doc.asset_class,
                "min_investment": doc.min_investment
            }
            for doc in retrieved_docs
        ]
        
        # Get investor dict
        investor_dict = {
            "age": investor_profile.age,
            "risk_tolerance": investor_profile.risk_tolerance,
            "investment_experience": investor_profile.investment_experience,
            "time_horizon_years": investor_profile.time_horizon_years,
            "net_worth": investor_profile.net_worth
        }
        
        # Generate recommendation
        llm_response = self.llm.generate_recommendation(
            investor_profile=investor_dict,
            user_query=user_query,
            retrieved_products=products_for_llm,
            constraints=constraints
        )
        
        print(f"   Recommended products: {llm_response['recommended_products']}")
        print(f"   Allocations: {llm_response['allocation_percentages']}")
        
        # Convert to PortfolioRecommendation
        recommendation = PortfolioRecommendation(
            investor_id=investor_profile.investor_id,
            recommended_products=llm_response['recommended_products'],
            allocation_percentages=llm_response['allocation_percentages'],
            rationale=llm_response['rationale']
        )
        
        # STEP 4: POST-VALIDATION (Tier 2 Safety - Fixed Swapped Argument Bug)
        print("\n🛡️  STEP 4: Post-generation validation (Phase 1)...")
        
        if self.validator:
            validation_result = self.validator.validate_recommendation(
                investor_profile,      # Swapped bug fixed here! It expects investor_profile first.
                recommendation
            )
            print(f"   Validation: {'✅ APPROVED' if validation_result.is_suitable else '❌ REJECTED'}")
            print(f"   Portfolio risk: {validation_result.overall_portfolio_risk:.2f}/5.0")
            
            if validation_result.risk_violations:
                print(f"   Violations: {len(validation_result.risk_violations)}")
                for violation in validation_result.risk_violations[:3]:
                    print(f"      - {violation}")
        else:
            validation_result = SuitabilityResult(
                is_suitable=True,
                risk_violations=[],
                compliance_notes=["Validation skipped - no validator loaded"],
                approved_products=recommendation.recommended_products,
                rejected_products=[],
                overall_portfolio_risk=0.0
            )
        
        # FINAL STATUS
        final_status = "APPROVED" if validation_result.is_suitable else "REJECTED"
        
        print(f"\n{'='*70}")
        print(f"📊 FINAL STATUS: {final_status}")
        print(f"{'='*70}\n")
        
        return {
            "investor_profile": investor_profile,
            "query": user_query,
            "retrieved_products": retrieved_docs,
            "llm_recommendation": recommendation,
            "validation_result": validation_result,
            "final_status": final_status
        }
    
    def _create_rejection_response(
        self,
        investor_profile: InvestorProfile,
        reason: str,
        retrieved_products: List
    ) -> Dict:
        """Create rejection response"""
        
        recommendation = PortfolioRecommendation(
            recommended_products=[],
            allocation_percentages=[],
            rationale=reason
        )
        
        validation_result = SuitabilityResult(
            is_suitable=False,
            risk_violations=[reason],
            compliance_notes=[reason],
            approved_products=[],
            rejected_products=[],
            overall_portfolio_risk=0.0
        )
        
        return {
            "investor_profile": investor_profile,
            "query": "",
            "retrieved_products": retrieved_products,
            "llm_recommendation": recommendation,
            "validation_result": validation_result,
            "final_status": "REJECTED"
        }
