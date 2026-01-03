"""
Fireworks AI LLM - Real Language Model for Production
Uses Fireworks AI's fast inference API for recommendations

Fireworks AI provides:
- Fast inference (< 1 second)
- Multiple models (Llama, Mixtral, etc.)
- Cost-effective pricing
- High quality outputs
"""
import requests
import os
import json
from typing import Dict, List, Optional

class FireworksLLM:
    """
    Real LLM using Fireworks AI API
    
    Fireworks offers fast inference for open-source models
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    ):
        """
        Initialize Fireworks LLM
        
        Args:
            api_key: Fireworks AI API key (get from https://fireworks.ai/)
            model: Model to use (default: Llama 3.1 70B)
            
        Available models:
        - accounts/fireworks/models/llama-v3p1-70b-instruct (recommended)
        - accounts/fireworks/models/llama-v3p1-8b-instruct (faster, cheaper)
        - accounts/fireworks/models/mixtral-8x7b-instruct
        """
        self.api_key = api_key or os.getenv('FIREWORKS_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Fireworks API key required! Get one at https://fireworks.ai/ and set:\n"
                "  export FIREWORKS_API_KEY='your-key-here'\n"
                "  or pass api_key parameter"
            )
        
        self.model = model
        self.api_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        
        print(f"✅ Fireworks AI LLM initialized")
        print(f"   Model: {self.model}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 - 1.0)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Fireworks API error: {response.status_code} - {response.text}")
            
            data = response.json()
            
            # Extract generated text
            generated_text = data['choices'][0]['message']['content']
            
            return generated_text
            
        except Exception as e:
            print(f"❌ Fireworks LLM error: {e}")
            raise
    
    def generate_recommendation(
        self,
        investor_profile: Dict,
        user_query: str,
        retrieved_products: List[Dict],
        constraints: List[str]
    ) -> Dict:
        """
        Generate investment recommendation
        
        Args:
            investor_profile: Investor information
            user_query: User's investment query
            retrieved_products: Products retrieved from vector search
            constraints: Safety constraints from validator
            
        Returns:
            Dictionary with recommendation
        """
        # Build prompt
        prompt = self._build_recommendation_prompt(
            investor_profile,
            user_query,
            retrieved_products,
            constraints
        )
        
        # Generate response
        response_text = self.generate(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )
        
        # Parse response
        recommendation = self._parse_recommendation(response_text, retrieved_products)
        
        return recommendation
    
    def _build_recommendation_prompt(
        self,
        investor_profile: Dict,
        user_query: str,
        retrieved_products: List[Dict],
        constraints: List[str]
    ) -> str:
        """Build structured prompt for investment recommendation"""
        
        # Format investor info
        investor_info = f"""
Investor Profile:
- Age: {investor_profile.get('age')}
- Risk Tolerance: {investor_profile.get('risk_tolerance')}/5
- Experience: {investor_profile.get('investment_experience')}
- Time Horizon: {investor_profile.get('time_horizon_years')} years
- Net Worth: ${investor_profile.get('net_worth'):,.0f}
"""
        
        # Format products
        products_info = "Available Products:\n"
        for i, product in enumerate(retrieved_products, 1):
            products_info += f"\n{i}. {product.get('ticker')} - {product.get('name')}\n"
            products_info += f"   Risk Score: {product.get('risk_score')}/5\n"
            products_info += f"   Asset Class: {product.get('asset_class')}\n"
            products_info += f"   Min Investment: ${product.get('min_investment'):,.0f}\n"
        
        # Format constraints
        constraints_info = "\nSafety Constraints (MUST FOLLOW):\n"
        for constraint in constraints:
            constraints_info += f"- {constraint}\n"
        
        # Build complete prompt
        prompt = f"""You are a professional investment advisor. Provide a recommendation for this investor.

{investor_info}

User Query: {user_query}

{products_info}

{constraints_info}

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
    
    def _parse_recommendation(self, response_text: str, retrieved_products: List[Dict]) -> Dict:
        """
        Parse LLM response into structured recommendation
        
        Args:
            response_text: Raw LLM response
            retrieved_products: Available products
            
        Returns:
            Parsed recommendation dictionary
        """
        try:
            # Try to extract JSON from response
            # Sometimes LLM adds extra text before/after JSON
            
            # Find JSON block
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start_idx:end_idx]
            recommendation = json.loads(json_str)
            
            # Validate structure
            if 'recommended_products' not in recommendation:
                raise ValueError("Missing 'recommended_products' field")
            
            if 'allocation_percentages' not in recommendation:
                raise ValueError("Missing 'allocation_percentages' field")
            
            if 'rationale' not in recommendation:
                recommendation['rationale'] = "Investment recommendation based on profile analysis"
            
            # Ensure allocations sum to 100
            allocations = recommendation['allocation_percentages']
            total = sum(allocations)
            if total != 100:
                # Normalize to 100
                recommendation['allocation_percentages'] = [
                    round(a * 100 / total, 1) for a in allocations
                ]
            
            return recommendation
            
        except Exception as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
            print(f"   Response: {response_text}")
            
            # Fallback: simple recommendation
            available_tickers = [p.get('ticker') for p in retrieved_products[:2]]
            return {
                "recommended_products": available_tickers,
                "allocation_percentages": [60, 40] if len(available_tickers) >= 2 else [100],
                "rationale": "Balanced portfolio recommendation based on investor profile"
            }


class MockLLM:
    """
    Fallback mock LLM for testing without API key
    (Keep this for backward compatibility)
    """
    
    def __init__(self):
        """Initialize mock LLM"""
        print(f"⚠️  Using MockLLM (for testing only)")
        print(f"   For production, use FireworksLLM with real API key")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""
        return '{"recommended_products": ["VTI", "BND"], "allocation_percentages": [60, 40], "rationale": "Mock recommendation"}'
    
    def generate_recommendation(
        self,
        investor_profile: Dict,
        user_query: str,
        retrieved_products: List[Dict],
        constraints: List[str]
    ) -> Dict:
        """Generate mock recommendation"""
        available_tickers = [p.get('ticker') for p in retrieved_products[:2]]
        
        return {
            "recommended_products": available_tickers,
            "allocation_percentages": [60, 40] if len(available_tickers) >= 2 else [100],
            "rationale": "Mock recommendation based on investor profile and risk tolerance"
        }


def get_llm(use_real: bool = True) -> FireworksLLM | MockLLM:
    """
    Get LLM instance (real or mock)
    
    Args:
        use_real: If True, try to use Fireworks AI (requires API key)
                 If False, use MockLLM
    
    Returns:
        FireworksLLM or MockLLM instance
    """
    if use_real:
        try:
            return FireworksLLM()
        except ValueError as e:
            print(f"⚠️  {e}")
            print(f"⚠️  Falling back to MockLLM")
            return MockLLM()
    else:
        return MockLLM()


# =========================================================================
# TESTING SCRIPT
# =========================================================================

if __name__ == "__main__":
    """
    Test Fireworks LLM
    
    Usage:
        export FIREWORKS_API_KEY='your-key-here'
        python fireworks_llm.py
    """
    print("\n🧪 TESTING FIREWORKS LLM\n")
    
    try:
        print("🤖 Testing Fireworks AI LLM:")
        llm = FireworksLLM()
        
        # Test simple generation
        prompt = "Explain what a stock market index fund is in one sentence."
        response = llm.generate(prompt, max_tokens=100)
        
        print(f"✅ Simple generation test:")
        print(f"   Prompt: {prompt}")
        print(f"   Response: {response}\n")
        
        # Test investment recommendation
        print("📊 Testing investment recommendation:")
        
        investor_profile = {
            "age": 30,
            "risk_tolerance": 4,
            "investment_experience": "intermediate",
            "time_horizon_years": 30,
            "net_worth": 200000
        }
        
        retrieved_products = [
            {
                "ticker": "VTI",
                "name": "Vanguard Total Stock Market",
                "risk_score": 3,
                "asset_class": "equity",
                "min_investment": 1000
            },
            {
                "ticker": "BND",
                "name": "Vanguard Total Bond Market",
                "risk_score": 2,
                "asset_class": "fixed_income",
                "min_investment": 1000
            }
        ]
        
        constraints = [
            "Risk score must not exceed investor's risk tolerance (4/5)",
            "Must diversify across at least 2 products",
            "Max 60% in any single product"
        ]
        
        recommendation = llm.generate_recommendation(
            investor_profile=investor_profile,
            user_query="I want growth with some stability",
            retrieved_products=retrieved_products,
            constraints=constraints
        )
        
        print(f"✅ Investment recommendation generated:")
        print(f"   Products: {recommendation['recommended_products']}")
        print(f"   Allocations: {recommendation['allocation_percentages']}")
        print(f"   Rationale: {recommendation['rationale']}")
        
        print("\n✅ FIREWORKS LLM TESTS COMPLETE!\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTo use Fireworks AI:")
        print("1. Get API key from https://fireworks.ai/")
        print("2. Set environment variable:")
        print("   export FIREWORKS_API_KEY='your-key-here'")
        print("3. Run this script again")
