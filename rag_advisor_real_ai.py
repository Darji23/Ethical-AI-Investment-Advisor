"""
RAG Investment Advisor with Real AI Models
Uses Jina AI embeddings + Fireworks AI LLM

This is the production-ready version with real AI services!
"""
from typing import Dict, List
import numpy as np

from schemas import InvestorProfile, PortfolioRecommendation, SuitabilityResult, Product
from vector_store import VectorStore, ProductDocument
from context_injector import ContextInjector
from suitability_validator_expanded import SuitabilityValidator

# Import real AI models
try:
    from jina_embedder import JinaEmbedder, MockEmbedder, get_embedder
    JINA_AVAILABLE = True
except ImportError:
    JINA_AVAILABLE = False
    print("⚠️  jina_embedder not found")

try:
    from fireworks_llm import FireworksLLM, MockLLM, get_llm
    FIREWORKS_AVAILABLE = True
except ImportError:
    FIREWORKS_AVAILABLE = False
    print("⚠️  fireworks_llm not found")


class RAGInvestmentAdvisorRealAI:
    """
    Complete RAG Pipeline with Real AI Models
    
    Flow:
    1. Embed user query (Jina AI)
    2. Retrieve relevant products (FAISS)
    3. Filter by risk/age constraints (Validator)
    4. Generate recommendation (Fireworks AI)
    5. Final validation (Validator)
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
        
        Args:
            vector_store: FAISS vector store with product embeddings
            embedder: Jina or Mock embedder (auto-detected if None)
            llm: Fireworks or Mock LLM (auto-detected if None)
            validator: Phase 1 validator
            use_real_ai: If True, try to use real AI (requires API keys)
        """
        self.vector_store = vector_store
        self.validator = validator
        self.context_injector = ContextInjector()
        
        # Auto-detect embedder
        if embedder is None:
            if use_real_ai and JINA_AVAILABLE:
                try:
                    self.embedder = get_embedder(use_real=True)
                    print("✅ Using Jina AI embeddings")
                except:
                    self.embedder = get_embedder(use_real=False)
                    print("⚠️  Falling back to MockEmbedder")
            else:
                # For testing - use mock with matching dimension
                class MockEmbedder:
                    def __init__(self):
                        self.embedding_dim = vector_store.embedding_dim
                    
                    def embed_query(self, text: str) -> np.ndarray:
                        return np.random.randn(self.embedding_dim).astype(np.float32)
                
                self.embedder = MockEmbedder()
                print("⚠️  Using MockEmbedder")
        else:
            self.embedder = embedder
        
        # Auto-detect LLM
        if llm is None:
            if use_real_ai and FIREWORKS_AVAILABLE:
                try:
                    self.llm = get_llm(use_real=True)
                    print("✅ Using Fireworks AI LLM")
                except:
                    self.llm = get_llm(use_real=False)
                    print("⚠️  Falling back to MockLLM")
            else:
                # For testing - use mock
                class MockLLM:
                    def generate_recommendation(self, investor_profile, user_query, retrieved_products, constraints):
                        available_tickers = [p['ticker'] for p in retrieved_products[:2]]
                        return {
                            "recommended_products": available_tickers,
                            "allocation_percentages": [60, 40] if len(available_tickers) >= 2 else [100],
                            "rationale": "Mock recommendation based on investor profile"
                        }
                
                self.llm = MockLLM()
                print("⚠️  Using MockLLM")
        else:
            self.llm = llm
        
        print(f"\n{'='*70}")
        print("🤖 RAG Investment Advisor Initialized (Real AI)")
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
        Generate investment advice using RAG + Validation
        
        Args:
            investor_profile: Investor information
            user_query: User's investment query
            top_k: Number of products to retrieve
            
        Returns:
            Dictionary with recommendation and validation results
        """
        print(f"\n{'─'*70}")
        print(f"🎯 Processing request for {investor_profile.investor_id}")
        print(f"   Query: {user_query}")
        print(f"{'─'*70}\n")
        
        # Step 1: TIER 1 FILTERING (Pre-retrieval risk filtering)
        print("🛡️  TIER 1: Pre-retrieval risk filtering...")
        max_risk = self.validator.get_max_risk_for_investor(investor_profile) if self.validator else 5
        
        eligible_products = [
            doc for doc in self.vector_store.documents
            if doc.risk_score <= max_risk
        ]
        
        print(f"   Max risk allowed: {max_risk}")
        print(f"   Eligible products: {len(eligible_products)}/{len(self.vector_store.documents)}")
        
        if not eligible_products:
            print("   ❌ No eligible products found!")
            return self._create_rejection_response(
                investor_profile,
                "No products match investor's risk profile",
                []
            )
        
        # Step 2: SEMANTIC SEARCH (Retrieve relevant products)
        print("\n🔍 STEP 2: Semantic search with Jina embeddings...")
        
        # Embed query using Jina AI
        query_embedding = self.embedder.embed_query(user_query)
        
        # Search in eligible products only
        eligible_indices = [
            self.vector_store.documents.index(doc)
            for doc in eligible_products
        ]
        
        # Get embeddings for eligible products
        eligible_embeddings = np.array([
            self.vector_store.index.reconstruct(i)
            for i in eligible_indices
        ])
        
        # Calculate similarities
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
        
        # Step 3: GENERATE RECOMMENDATION (Fireworks AI)
        print("\n🤖 STEP 3: Generating recommendation with Fireworks AI...")
        
        # Build constraints from validator
        constraints = self.context_injector.get_constraints(investor_profile)
        
        # Convert docs to dict format for LLM
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
        
        # Get investor profile dict
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
            retrieval_method="rag_with_jina_fireworks"
        )
        
        # Step 4: TIER 2 VALIDATION (Post-generation validation)
        print("\n🛡️  TIER 2: Post-generation validation...")
        
        if self.validator:
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
        else:
            # No validator - auto-approve
            validation_result = SuitabilityResult(
                is_suitable=True,
                recommended_products=recommendation.recommended_products,
                approved_products=recommendation.recommended_products,
                rejected_products=[],
                overall_portfolio_risk=0,
                risk_violations=[],
                compliance_notes=["Validation skipped - no validator provided"]
            )
        
        # Determine final status
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
        """Create rejection response when no suitable products found"""
        
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


# Backward compatibility alias
RAGInvestmentAdvisor = RAGInvestmentAdvisorRealAI
