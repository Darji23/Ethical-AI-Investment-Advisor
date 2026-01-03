"""
RAG Investment Advisor (Phase 2)
Complete pipeline: Jina Embeddings → FAISS → Fireworks LLM → Validator

Integrates:
- Phase 1: Suitability Validator
- Phase 2: RAG pipeline with real AI
"""
from typing import Dict, List
import numpy as np

from schemas import InvestorProfile, PortfolioRecommendation, SuitabilityResult, Product
from vector_store import VectorStore, ProductDocument
from context_injector import ContextInjector
from suitability_validator_expanded import SuitabilityValidator

# Import real AI models
from jina_embedder import JinaEmbedder
from fireworks_llm import FireworksLLM


class RAGInvestmentAdvisor:
    """
    Complete RAG Pipeline with Real AI
    
    Flow:
    1. User query → Jina AI embedding
    2. FAISS semantic search → Retrieve relevant products
    3. Pre-filter by risk (Phase 1 validator)
    4. Build context-aware prompt
    5. Fireworks AI generates recommendation
    6. Post-validation (Phase 1 validator)
    7. Return final recommendation
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: JinaEmbedder,
        llm: FireworksLLM,
        validator: SuitabilityValidator
    ):
        """
        Initialize RAG advisor
        
        Args:
            vector_store: FAISS vector store with product embeddings
            embedder: Jina AI embedder
            llm: Fireworks AI LLM
            validator: Phase 1 suitability validator
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        self.validator = validator
        self.context_injector = ContextInjector()
        
        print(f"\n{'='*70}")
        print("🤖 RAG Investment Advisor Initialized")
        print(f"{'='*70}")
        print(f"Embedder: {type(self.embedder).__name__}")
        print(f"LLM: {type(self.llm).__name__}")
        print(f"Vector Store: {len(self.vector_store.documents)} products indexed")
        print(f"Validator: Enabled (Phase 1)")
        print(f"{'='*70}\n")
    
    def get_advice(
        self,
        investor_profile: InvestorProfile,
        user_query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Generate investment advice using complete RAG pipeline
        
        Args:
            investor_profile: Investor information
            user_query: User's investment query
            top_k: Number of products to retrieve
            
        Returns:
            Dictionary with:
            - investor_profile
            - query
            - retrieved_products
            - llm_recommendation
            - validation_result
            - final_status (APPROVED/REJECTED)
        """
        print(f"\n{'─'*70}")
        print(f"🎯 Processing request for {investor_profile.investor_id}")
        print(f"   Query: {user_query}")
        print(f"{'─'*70}\n")
        
        # STEP 1: PRE-FILTERING (Tier 1 Safety)
        print("🛡️  STEP 1: Pre-retrieval risk filtering...")
        max_risk = self.validator.get_max_risk_for_investor(investor_profile)
        
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
            np.dot(query_embedding, emb) / (norm(query_embedding) * norm(emb))
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
            rationale=llm_response['rationale'],
            retrieval_method="rag_jina_fireworks"
        )
        
        # STEP 4: POST-VALIDATION (Tier 2 Safety)
        print("\n🛡️  STEP 4: Post-generation validation (Phase 1)...")
        
        validation_result = self.validator.validate_recommendation(
            recommendation,
            investor_profile
        )
        
        print(f"   Validation: {'✅ APPROVED' if validation_result.is_suitable else '❌ REJECTED'}")
        print(f"   Portfolio risk: {validation_result.overall_portfolio_risk:.2f}/5.0")
        
        if validation_result.risk_violations:
            print(f"   Violations: {len(validation_result.risk_violations)}")
            for violation in validation_result.risk_violations[:3]:
                print(f"      - {violation}")
        
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
            investor_id=investor_profile.investor_id,
            recommended_products=[],
            allocation_percentages=[],
            rationale=reason,
            retrieval_method="rag_filtered"
        )
        
        validation_result = SuitabilityResult(
            is_suitable=False,
            recommended_products=[],
            approved_products=[],
            rejected_products=[],
            overall_portfolio_risk=0,
            risk_violations=[reason],
            compliance_notes=[reason]
        )
        
        return {
            "investor_profile": investor_profile,
            "query": "",
            "retrieved_products": retrieved_products,
            "llm_recommendation": recommendation,
            "validation_result": validation_result,
            "final_status": "REJECTED"
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test RAG advisor with real AI"""
    
    print("\n🧪 TESTING RAG ADVISOR WITH REAL AI\n")
    
    import json
    from schemas import InvestorProfile, Product
    
    # Load data
    print("📦 Loading data...")
    
    with open('product_master.json', 'r') as f:
        products_data = json.load(f)
        products = [Product(**p) for p in products_data]
    
    print(f"✅ Loaded {len(products)} products")
    
    # Initialize components
    print("\n🤖 Initializing AI components...")
    
    embedder = JinaEmbedder()
    llm = FireworksLLM()
    validator = SuitabilityValidator(products)
    
    # Load vector store
    vector_store = VectorStore(embedding_dim=1024)
    vector_store.load('product_vector_store')
    
    # Create advisor
    advisor = RAGInvestmentAdvisor(
        vector_store=vector_store,
        embedder=embedder,
        llm=llm,
        validator=validator
    )
    
    # Test with sample investor
    print("\n" + "="*70)
    print("TEST CASE 1: Young aggressive investor")
    print("="*70)
    
    investor1 = InvestorProfile(
        investor_id="TEST001",
        age=30,
        risk_tolerance=5,
        annual_income=120000,
        net_worth=250000,
        investment_experience="intermediate",
        time_horizon_years=35
    )
    
    result1 = advisor.get_advice(
        investor_profile=investor1,
        user_query="I want aggressive growth with tech exposure",
        top_k=5
    )
    
    print("\nRESULT:")
    print(f"Status: {result1['final_status']}")
    print(f"Products: {result1['llm_recommendation'].recommended_products}")
    print(f"Allocations: {result1['llm_recommendation'].allocation_percentages}")
    print(f"Rationale: {result1['llm_recommendation'].rationale}")
    
    # Test with elderly conservative investor
    print("\n" + "="*70)
    print("TEST CASE 2: Elderly conservative investor")
    print("="*70)
    
    investor2 = InvestorProfile(
        investor_id="TEST002",
        age=72,
        risk_tolerance=2,
        annual_income=60000,
        net_worth=500000,
        investment_experience="beginner",
        time_horizon_years=5
    )
    
    result2 = advisor.get_advice(
        investor_profile=investor2,
        user_query="I need safe stable income for retirement",
        top_k=5
    )
    
    print("\nRESULT:")
    print(f"Status: {result2['final_status']}")
    print(f"Products: {result2['llm_recommendation'].recommended_products}")
    print(f"Allocations: {result2['llm_recommendation'].allocation_percentages}")
    print(f"Rationale: {result2['llm_recommendation'].rationale}")
    
    print("\n✅ RAG ADVISOR TESTS COMPLETE!\n")
