"""
Context Injector for RAG (Phase 2)
Builds prompts with investor context and safety constraints
"""
from typing import List, Dict
from schemas import InvestorProfile, Product

class ContextInjector:
    """
    Injects investor context and safety constraints into prompts
    Ensures LLM has all necessary information for safe recommendations
    """
    
    def build_prompt(
        self,
        investor_profile: InvestorProfile,
        user_query: str,
        retrieved_products: List[Product],
        max_risk: int = 5
    ) -> str:
        """
        Build comprehensive prompt for LLM
        
        Args:
            investor_profile: Investor information
            user_query: User's investment question
            retrieved_products: Products retrieved from vector search
            max_risk: Maximum allowed risk level
            
        Returns:
            Complete prompt string
        """
        # Build investor context
        investor_context = self._build_investor_context(investor_profile)
        
        # Build product list
        product_list = self._build_product_list(retrieved_products)
        
        # Build safety constraints
        constraints = self._build_constraints(investor_profile, max_risk)
        
        # Combine into full prompt
        prompt = f"""You are a professional investment advisor. Provide a recommendation for this investor.

{investor_context}

USER QUERY: {user_query}

AVAILABLE PRODUCTS:
{product_list}

SAFETY CONSTRAINTS (MUST FOLLOW):
{constraints}

INSTRUCTIONS:
1. Select 2-3 products from the list above that match the investor's profile
2. Provide allocation percentages (must sum to 100%)
3. Explain your rationale in 2-3 sentences
4. Consider the investor's age, risk tolerance, and experience
5. STRICTLY follow all safety constraints

OUTPUT FORMAT (respond ONLY with valid JSON):
{{
    "recommended_products": ["TICKER1", "TICKER2"],
    "allocation_percentages": [60, 40],
    "rationale": "Brief explanation here"
}}"""
        
        return prompt
    
    def _build_investor_context(self, investor: InvestorProfile) -> str:
        """Build investor profile section"""
        return f"""INVESTOR PROFILE:
- ID: {investor.investor_id}
- Age: {investor.age} years old
- Risk Tolerance: {investor.risk_tolerance}/5
- Experience: {investor.investment_experience}
- Time Horizon: {investor.time_horizon_years} years
- Annual Income: ${investor.annual_income:,.0f}
- Net Worth: ${investor.net_worth:,.0f}"""
    
    def _build_product_list(self, products: List[Product]) -> str:
        """Build formatted product list"""
        product_text = ""
        for i, product in enumerate(products, 1):
            product_text += f"\n{i}. {product.ticker} - {product.name}\n"
            product_text += f"   Risk Score: {product.risk_score}/5\n"
            product_text += f"   Asset Class: {product.asset_class}\n"
            product_text += f"   Min Investment: ${product.min_investment:,.0f}\n"
        return product_text
    
    def _build_constraints(self, investor: InvestorProfile, max_risk: int) -> str:
        """Build safety constraints based on investor profile"""
        constraints = []
        
        # Age-based constraints
        if investor.age >= 70:
            constraints.append(f"- Investor is elderly ({investor.age} years old) - AVOID high-risk products (risk ≥ 4)")
            constraints.append("- Prioritize capital preservation and stable income")
        elif investor.age < 25:
            constraints.append(f"- Investor is young ({investor.age} years old) - Can handle moderate volatility")
        
        # Risk tolerance constraints
        constraints.append(f"- Maximum risk score allowed: {max_risk}/5")
        constraints.append(f"- Investor's risk tolerance: {investor.risk_tolerance}/5")
        
        # Experience-based constraints
        if investor.investment_experience == "beginner":
            constraints.append("- Investor is a BEGINNER - Avoid complex products (options, crypto, leveraged ETFs)")
            constraints.append("- Recommend simple, well-diversified index funds")
        elif investor.investment_experience == "intermediate":
            constraints.append("- Investor has INTERMEDIATE experience - Can handle moderate complexity")
        else:  # advanced
            constraints.append("- Investor is ADVANCED - Can evaluate complex strategies")
        
        # Time horizon constraints
        if investor.time_horizon_years < 5:
            constraints.append(f"- SHORT time horizon ({investor.time_horizon_years} years) - Favor lower-risk, liquid investments")
        elif investor.time_horizon_years >= 20:
            constraints.append(f"- LONG time horizon ({investor.time_horizon_years} years) - Can tolerate short-term volatility")
        
        # Portfolio construction rules
        constraints.append("- MUST recommend at least 2 products for diversification")
        constraints.append("- No single product should exceed 60% allocation")
        constraints.append("- Allocation percentages MUST sum to exactly 100%")
        
        return "\n".join(constraints)
    
    def get_constraints(self, investor: InvestorProfile) -> List[str]:
        """
        Get list of constraints for investor
        Used by RAG advisor for filtering
        
        Args:
            investor: InvestorProfile object
            
        Returns:
            List of constraint strings
        """
        constraints = []
        
        # Age constraints
        if investor.age >= 70:
            constraints.append(f"Elderly investor ({investor.age} years) - avoid high-risk products")
        elif investor.age < 25:
            constraints.append(f"Young investor ({investor.age} years) - can handle volatility")
        
        # Risk constraints
        constraints.append(f"Risk tolerance: {investor.risk_tolerance}/5")
        
        # Experience constraints
        if investor.investment_experience == "beginner":
            constraints.append("Beginner - avoid complex products (options, crypto, leverage)")
        elif investor.investment_experience == "intermediate":
            constraints.append("Intermediate experience - moderate complexity OK")
        else:
            constraints.append("Advanced - can handle complex strategies")
        
        # Time horizon
        if investor.time_horizon_years < 5:
            constraints.append(f"Short horizon ({investor.time_horizon_years} years) - favor liquidity")
        elif investor.time_horizon_years >= 20:
            constraints.append(f"Long horizon ({investor.time_horizon_years} years) - growth focus")
        
        # Portfolio rules
        constraints.append("Must diversify (2-3 products minimum)")
        constraints.append("Max 60% allocation per product")
        
        return constraints


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test context injector"""
    
    print("\n🧪 TESTING CONTEXT INJECTOR\n")
    
    from schemas import InvestorProfile, Product
    
    # Create test investor
    investor = InvestorProfile(
        investor_id="TEST001",
        age=30,
        risk_tolerance=4,
        annual_income=95000,
        net_worth=200000,
        investment_experience="intermediate",
        time_horizon_years=30
    )
    
    # Create test products
    products = [
        Product(
            ticker="VTI",
            name="Vanguard Total Stock Market",
            risk_score=3,
            asset_class="equity",
            min_investment=1000
        ),
        Product(
            ticker="BND",
            name="Vanguard Total Bond Market",
            risk_score=2,
            asset_class="fixed_income",
            min_investment=1000
        )
    ]
    
    # Test context injector
    injector = ContextInjector()
    
    prompt = injector.build_prompt(
        investor_profile=investor,
        user_query="I want balanced growth with some stability",
        retrieved_products=products,
        max_risk=4
    )
    
    print("="*70)
    print("GENERATED PROMPT:")
    print("="*70)
    print(prompt)
    print("="*70)
    
    # Test constraints
    constraints = injector.get_constraints(investor)
    print("\nCONSTRAINTS:")
    for i, constraint in enumerate(constraints, 1):
        print(f"{i}. {constraint}")
    
    print("\n✅ CONTEXT INJECTOR TEST COMPLETE!\n")
