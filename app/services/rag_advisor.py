"""
RAG Investment Advisor
Complete pipeline: Jina Embeddings → FAISS → Fireworks LLM → Validator
Falls back to MockEmbedder / the local HF model if API keys are missing.
Resolves parameter swapping bugs in post-validation.
"""
from typing import Dict, List

from app.core.config import RETRIEVAL_TOP_K
import numpy as np

from app.schemas.schemas import InvestorProfile, PortfolioRecommendation, SuitabilityResult, Product
from app.services.vector_store import VectorStore, ProductDocument
from app.services.context_injector import ContextInjector
from app.services.suitability_validator import SuitabilityValidator

# Import real AI models & get_embedder/get_llm helpers
from app.services.jina_embedder import JinaEmbedder, MockEmbedder, get_embedder
from app.services.fireworks_llm import FireworksLLM, LocalLLM, get_llm

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
        if not self.uses_real_ai:
            print("⚠️  NOT the benchmarked system: real embedder/LLM unavailable "
                  "(check JINA_API_KEY and FIREWORKS_API_KEY). Expected 'JinaEmbedder' and 'FireworksLLM'.")
        print(f"{'='*70}\n")

    @property
    def uses_real_ai(self) -> bool:
        """True when the same embedder and LLM as the benchmark are active (not MockEmbedder / LocalLLM)."""
        return type(self.embedder).__name__ == "JinaEmbedder" and type(self.llm).__name__ == "FireworksLLM"
    
    def get_advice(
        self,
        investor_profile: InvestorProfile,
        user_query: str,
        top_k: int = RETRIEVAL_TOP_K
    ) -> Dict:
        """
        Generate investment advice using complete RAG pipeline
        """
        print(f"\n{'─'*70}")
        print(f"🎯 Processing request for {investor_profile.investor_id}")
        print(f"   Query: {user_query}")
        print(f"{'─'*70}\n")
        
        # STEP 1: PRE-FILTERING (Tier 1 Safety — exact consistency with Tier 2)
        print("🛡️  STEP 1: Pre-retrieval risk filtering...")
        max_risk = self.validator.get_max_risk_for_investor(investor_profile) if self.validator else 5
        if self.validator:
            suitable_set = set(self.validator.get_suitable_products(investor_profile))
            eligible_products = [
                doc for doc in self.vector_store.documents
                if doc.ticker in suitable_set
            ]
        else:
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

        # Enrich query with investor's investment_objective so retrieval aligns with mandate
        obj = getattr(investor_profile, "investment_objective", "balanced")
        objective_hints = {
            "preservation": "capital preservation treasury short-term low volatility",
            "income": "income bond dividend yield fixed income",
            "balanced": "balanced diversified blend",
            "growth": "growth equity capital appreciation",
            "speculation": "high growth thematic aggressive",
        }
        search_query = f"{user_query} {objective_hints.get(obj, '')}".strip()

        # Generate query embedding
        query_embedding = self.embedder.embed_query(search_query)
        
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
        
        # STEP 3 & 4: GENERATE + VALIDATE WITH RE-PROMPT LOOP & SAFE FALLBACK
        print("\n🤖 STEP 3: Generating recommendation with Fireworks AI...")
        constraints = self.context_injector.get_constraints(investor_profile)

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

        # Get investor dict — name intentionally excluded (counterfactual fairness)
        investor_dict = {
            "age": investor_profile.age,
            "risk_tolerance": investor_profile.risk_tolerance,
            "investment_experience": investor_profile.investment_experience,
            "time_horizon_years": investor_profile.time_horizon_years,
            "investable_assets": getattr(investor_profile, "investable_assets", investor_profile.net_worth),
            "investment_objective": getattr(investor_profile, "investment_objective", "balanced"),
            "liquidity_needs": getattr(investor_profile, "liquidity_needs", "medium"),
        }

        catalog_version = (
            next((p.catalog_version for p in self.validator.products.values()), "catalog_v2")
            if self.validator else "catalog_v2"
        )

        feedback = ""
        llm_response = {}
        recommendation = None
        validation_result = None
        max_attempts = 2
        fallback_used = False
        intervened_rules: List[str] = []
        recovery_paths: List[str] = []
        llm_failures: List[str] = []
        total_llm_latency_s = 0.0

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                recovery_paths.append("reprompt")
            try:
                llm_response = self.llm.generate_recommendation(
                    investor_profile=investor_dict,
                    user_query=user_query,
                    retrieved_products=products_for_llm,
                    constraints=constraints,
                    temperature=0.0,
                    feedback=feedback,
                )
            except TimeoutError as llm_err:
                llm_failures.append("timeout")
                llm_response = {"parse_error": f"LLM timeout: {llm_err}", "failure_type": "timeout",
                                "recommended_products": [], "allocation_percentages": []}
                feedback = f"Previous call timed out ({llm_err}); retry cleanly."
                recovery_paths.append("parse_failure")
                continue
            except Exception as llm_err:
                llm_failures.append("api_error")
                llm_response = {"parse_error": f"LLM call failed: {llm_err}", "failure_type": "api_error",
                                "recommended_products": [], "allocation_percentages": []}
                feedback = f"Previous call errored ({llm_err}); retry cleanly."
                recovery_paths.append("parse_failure")
                continue

            total_llm_latency_s += float(llm_response.get("llm_latency_s", 0.0))
            if llm_response.get("failure_type"):
                llm_failures.append(llm_response["failure_type"])
            if llm_response.get("primary_failure"):
                llm_failures.append(llm_response["primary_failure"])

            print(f"   [Attempt {attempt}/{max_attempts}] Recommended: {llm_response.get('recommended_products', [])} {llm_response.get('allocation_percentages', [])}")

            # Never build/validate a portfolio from a failed parse or an explicit refusal
            if llm_response.get("parse_error"):
                recovery_paths.append("parse_failure")
                feedback = f"Previous JSON failed strict validation: {llm_response['parse_error']}. Return valid JSON with known tickers summing to 100%."
                continue

            if llm_response.get("refused"):
                # The candidates are already pre-filtered, so a refusal is worth one re-prompt; then fallback.
                recovery_paths.append("llm_refusal")
                feedback = (
                    f"You reported no suitable products ({llm_response.get('reason', '')}). "
                    "Every product listed above has already passed the suitability pre-filter; "
                    "choose 2-3 of them if any fit the constraints."
                )
                continue

            recommendation = PortfolioRecommendation(
                recommended_products=llm_response["recommended_products"],
                allocation_percentages=llm_response["allocation_percentages"],
                rationale=llm_response.get("rationale") or "LLM recommendation.",
            )

            if not self.validator:
                validation_result = SuitabilityResult(
                    is_suitable=True,
                    risk_violations=[],
                    compliance_notes=["Validation skipped - no validator loaded"],
                    approved_products=recommendation.recommended_products,
                    rejected_products=[],
                    overall_portfolio_risk=0.0,
                )
                break

            validation_result = self.validator.validate_recommendation(investor_profile, recommendation)
            for v in validation_result.risk_violations:
                code = v.split(":")[0].strip()
                intervened_rules.append(code)
                if code == "UNKNOWN_TICKER":
                    llm_failures.append("invented_ticker")
            if validation_result.is_suitable:
                break

            # Build concise rejection feedback for re-prompt attempt
            feedback = (
                f"Rejected tickers: {validation_result.rejected_products}. "
                f"Violations: {'; '.join(validation_result.risk_violations[:3])}. "
                f"Choose only from the available catalog products above."
            )

        # If partial approvals exist with rejections, rebalance approved products to 100% and re-validate
        if self.validator and validation_result and not validation_result.is_suitable and validation_result.approved_products and validation_result.rebalanced_allocations:
            recovery_paths.append("rebalance")
            reb_tickers = list(validation_result.rebalanced_allocations.keys())
            reb_allocs = list(validation_result.rebalanced_allocations.values())
            rebalanced_rec = PortfolioRecommendation(
                recommended_products=reb_tickers,
                allocation_percentages=reb_allocs,
                rationale=f"{recommendation.rationale} (Rebalanced to 100% after removing unsuitable products.)",
            )
            reb_val = self.validator.validate_recommendation(investor_profile, rebalanced_rec)
            if reb_val.is_suitable:
                recommendation = rebalanced_rec
                validation_result = reb_val

        # Safe deterministic fallback if LLM failed, JSON broke, or everything was rejected
        if self.validator and (validation_result is None or not validation_result.is_suitable):
            reason = llm_response.get("parse_error") or (llm_response.get("reason") if llm_response.get("refused") else None) or "; ".join((validation_result.risk_violations if validation_result else [])[:2]) or "all products rejected"
            fallback_rec = self.validator.build_safe_fallback_portfolio(investor_profile, reason=reason)
            if fallback_rec is None:
                return self._create_rejection_response(
                    investor_profile,
                    "No suitable recommendation available under current suitability rules; please refer to a human advisor.",
                    retrieved_docs,
                )
            recovery_paths.append("fallback")
            recommendation = fallback_rec
            validation_result = self.validator.validate_recommendation(investor_profile, recommendation)
            validation_result.fallback_used = True
            fallback_used = True

        audit_meta = {
            "rule_set_version": self.validator.rule_set_version if self.validator else "none",
            "catalog_version": catalog_version,
            "model_name": llm_response.get("model_name", type(self.llm).__name__),
            "model_tier": llm_response.get("model_tier", "local" if isinstance(self.llm, LocalLLM) else "primary"),
            "temperature": llm_response.get("temperature", 0.0),
            "prompt_hash": llm_response.get("prompt_hash", ""),
            "fallback_used": fallback_used,
            "intervened_rules": intervened_rules,
            "recovery_paths": recovery_paths,
            "llm_failures": llm_failures,
            "llm_latency_s": round(total_llm_latency_s, 4),
            "failure_type": llm_response.get("failure_type"),
            "llm_refused": bool(llm_response.get("refused")),
        }
        if validation_result:
            validation_result.audit_metadata.update(audit_meta)

        final_status = "APPROVED" if (validation_result and validation_result.is_suitable) else "REJECTED"

        print(f"\n{'='*70}")
        print(f"📊 FINAL STATUS: {final_status} (fallback_used={fallback_used})")
        print(f"{'='*70}\n")

        return {
            "investor_profile": investor_profile,
            "query": user_query,
            "retrieved_products": retrieved_docs,
            "llm_recommendation": recommendation,
            "validation_result": validation_result,
            "final_status": final_status,
            "audit_metadata": audit_meta,
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
