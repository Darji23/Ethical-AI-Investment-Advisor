"""
Fireworks AI LLM - Real Language Model with Automatic Dual-Tier Failover
Prime Model: accounts/fireworks/models/deepseek-v4p1-flash (High accuracy, ultra-fast, low cost)
Fallback Model: accounts/fireworks/models/glm-5p3-flash (Open-weights model)
Tertiary Fallback: MockLLM for offline mode
"""
import requests
import os
import json
import re
import hashlib
from typing import Dict, List, Optional
from app.core.config import FIREWORKS_API_KEY, FIREWORKS_MODEL, FIREWORKS_FALLBACK_MODEL
from app.schemas.schemas import PortfolioRecommendation

DEFAULT_PRIME_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"
DEFAULT_FALLBACK_MODEL = "accounts/fireworks/models/glm-5p3-flash"

class FireworksLLM:
    """
    Real LLM client for Fireworks AI with dual-tier failover and resilient parsing.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        fallback_model: str = None
    ):
        """
        Initialize Fireworks LLM
        """
        self.api_key = api_key or FIREWORKS_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Fireworks API key required! Get one at https://fireworks.ai/ and set:\n"
                "  export FIREWORKS_API_KEY='your-key-here'\n"
                "  or pass api_key parameter"
            )
        
        self.prime_model = model or FIREWORKS_MODEL or DEFAULT_PRIME_MODEL
        self.fallback_model = fallback_model or FIREWORKS_FALLBACK_MODEL or DEFAULT_FALLBACK_MODEL
        self.api_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.mock_fallback = MockLLM()
        
        print(f"✅ Fireworks AI LLM initialized")
        print(f"   Prime Model: {self.prime_model}")
        print(f"   Fallback Model: {self.fallback_model}")

    def _call_fireworks_api(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.2
    ) -> str:
        """Call Fireworks chat completion endpoint"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code} - {response.text}")
        
        data = response.json()
        choice = data['choices'][0]
        message = choice.get('message', {})
        content = message.get('content')
        
        # If content is empty (e.g. pure reasoning output), try reasoning_content
        if not content and message.get('reasoning_content'):
            content = message['reasoning_content']
            
        if not content:
            raise Exception("Empty message content returned by Fireworks API")
            
        return content

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2
    ) -> str:
        """
        Generate text with automatic failover between prime and fallback models
        """
        # 1. Try Prime Model
        try:
            return self._call_fireworks_api(self.prime_model, prompt, max_tokens, temperature)
        except Exception as prime_err:
            print(f"⚠️  Fireworks prime model '{self.prime_model}' error: {prime_err}")
            print(f"🔄 Switching to fallback model: '{self.fallback_model}'...")

        # 2. Try Fallback Model
        try:
            res = self._call_fireworks_api(self.fallback_model, prompt, max_tokens, temperature)
            print(f"✅ Generated output with fallback model '{self.fallback_model}'.")
            return res
        except Exception as fallback_err:
            print(f"⚠️  Fireworks fallback model '{self.fallback_model}' failed: {fallback_err}")
            print("🛡️  Falling back to local MockLLM...")
            
        # 3. Tertiary Local Fallback
        return self.mock_fallback.generate(prompt)

    def generate_recommendation(
        self,
        investor_profile: Dict,
        user_query: str,
        retrieved_products: List[Dict],
        constraints: List[str],
        temperature: float = 0.0,
        feedback: str = ""
    ) -> Dict:
        """
        Generate investment recommendation with failover, strict Pydantic validation, and replay metadata.
        """
        prompt = self._build_recommendation_prompt(
            investor_profile,
            user_query,
            retrieved_products,
            constraints,
            feedback=feedback
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        response_text = self.generate(
            prompt=prompt,
            max_tokens=4096,
            temperature=temperature
        )

        recommendation = self._parse_recommendation(response_text, retrieved_products)
        recommendation["prompt_hash"] = prompt_hash
        recommendation["model_name"] = self.prime_model
        recommendation["temperature"] = temperature
        return recommendation

    def _build_recommendation_prompt(
        self,
        investor_profile: Dict,
        user_query: str,
        retrieved_products: List[Dict],
        constraints: List[str],
        feedback: str = ""
    ) -> str:
        """Build structured prompt for investment recommendation"""
        
        investor_info = f"""
Investor Profile:
- Age: {investor_profile.get('age')}
- Risk Tolerance: {investor_profile.get('risk_tolerance')}/5
- Experience: {investor_profile.get('investment_experience')}
- Time Horizon: {investor_profile.get('time_horizon_years')} years
- Investable Assets: ${investor_profile.get('investable_assets', investor_profile.get('net_worth', 0)):,.0f}
- Investment Objective: {investor_profile.get('investment_objective', 'balanced')}
- Liquidity Needs: {investor_profile.get('liquidity_needs', 'medium')}
"""
        
        products_info = "Available Catalog Products:\n"
        for i, product in enumerate(retrieved_products, 1):
            products_info += f"\n{i}. {product.get('ticker')} - {product.get('name')}\n"
            products_info += f"   Risk Score: {product.get('risk_score')}/5\n"
            products_info += f"   Asset Class: {product.get('asset_class')}\n"
            products_info += f"   Min Investment: ${product.get('min_investment', 0):,.0f}\n"
        
        constraints_info = "\nRegulatory & Safety Constraints (MUST STRICTLY FOLLOW):\n"
        for constraint in constraints:
            constraints_info += f"- {constraint}\n"
        if feedback:
            constraints_info += f"\nPREVIOUS ATTEMPT REJECTED — FIX THESE ISSUES:\n{feedback}\n"
        
        prompt = f"""You are a certified, ethical investment advisor. Provide a personalized recommendation for this investor based strictly on the available products.

{investor_info}

User Query: {user_query}

{products_info}

{constraints_info}

INSTRUCTIONS:
1. Select 2-3 products from the list above that best match the investor's profile and query.
2. Provide allocation percentages that sum EXACTLY to 100%.
3. Ensure no single product allocation exceeds 60%.
4. Keep the portfolio risk within the investor's risk ceiling.
5. Provide a clear, professional rationale in 2-3 sentences.

CRITICAL: Respond ONLY with a valid JSON object. Do not include markdown code blocks or extra text:
{{
    "recommended_products": ["TICKER1", "TICKER2"],
    "allocation_percentages": [60, 40],
    "rationale": "Brief explanation of how the portfolio aligns with risk profile and time horizon."
}}"""
        
        return prompt

    def _parse_recommendation(self, response_text: str, retrieved_products: List[Dict]) -> Dict:
        """
        Parse LLM response into structured recommendation with automatic repair
        """
        try:
            cleaned = response_text.strip()
            # Remove markdown backticks if present
            if "```" in cleaned:
                cleaned = re.sub(r'```(?:json)?', '', cleaned).strip()
            
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}') + 1
            
            if start_idx == -1:
                raise ValueError("No JSON object found in response")
            
            if end_idx <= start_idx:
                # Attempt to repair unclosed JSON string
                repaired = cleaned[start_idx:]
                if not repaired.endswith('"'):
                    repaired += '"'
                repaired += '}'
                json_str = repaired
            else:
                json_str = cleaned[start_idx:end_idx]
            
            recommendation = json.loads(json_str)
            
            if 'recommended_products' not in recommendation or not recommendation['recommended_products']:
                raise ValueError("Missing 'recommended_products' field")
            
            if 'allocation_percentages' not in recommendation or not recommendation['allocation_percentages']:
                raise ValueError("Missing 'allocation_percentages' field")
            
            if 'rationale' not in recommendation or not recommendation['rationale']:
                recommendation['rationale'] = "Investment recommendation aligned with profile analysis."

            # Strict Pydantic validation (checks numeric allocations, 99.5-100.5 sum, length match)
            validated = PortfolioRecommendation(**recommendation)
            return {
                "recommended_products": [str(t).strip().upper() for t in validated.recommended_products],
                "allocation_percentages": [float(a) for a in validated.allocation_percentages],
                "rationale": validated.rationale,
                "parse_error": None,
            }

        except Exception as e:
            print(f"⚠️  Failed to parse/validate LLM response: {e}")
            print(f"   Raw Response: {str(response_text)[:200]}")

            available_tickers = [p.get('ticker') for p in retrieved_products[:2]]
            return {
                "recommended_products": available_tickers if available_tickers else ["SPY", "BND"],
                "allocation_percentages": [60.0, 40.0] if len(available_tickers) >= 2 else [100.0],
                "rationale": "Fallback recommendation due to LLM parse/validation failure.",
                "parse_error": str(e),
            }


class MockLLM:
    """
    Fallback mock LLM for testing without API key or offline mode
    """
    
    def __init__(self):
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        return '{"recommended_products": ["SPY", "BND"], "allocation_percentages": [60, 40], "rationale": "Balanced portfolio recommendation based on moderate risk tolerance."}'
    
    def generate_recommendation(
        self,
        investor_profile: Dict,
        user_query: str,
        retrieved_products: List[Dict],
        constraints: List[str],
        temperature: float = 0.0,
        feedback: str = ""
    ) -> Dict:
        available_tickers = [p.get('ticker') for p in retrieved_products[:2]]
        return {
            "recommended_products": available_tickers if available_tickers else ["SPY", "BND"],
            "allocation_percentages": [60.0, 40.0] if len(available_tickers) >= 2 else [100.0],
            "rationale": "Standard balanced diversification recommendation aligned with investor suitability constraints.",
            "prompt_hash": hashlib.sha256(user_query.encode("utf-8")).hexdigest()[:16],
            "model_name": "MockLLM",
            "temperature": temperature,
            "parse_error": None,
        }


def get_llm(use_real: bool = True) -> FireworksLLM | MockLLM:
    """
    Get LLM instance (real with automatic failover, or mock)
    """
    if use_real and FIREWORKS_API_KEY:
        try:
            return FireworksLLM()
        except Exception as e:
            print(f"⚠️  Could not initialize FireworksLLM: {e}")
            print(f"⚠️  Falling back to MockLLM")
            return MockLLM()
    else:
        if use_real and not FIREWORKS_API_KEY:
            print("ℹ️  No FIREWORKS_API_KEY detected. Running with MockLLM.")
        return MockLLM()
