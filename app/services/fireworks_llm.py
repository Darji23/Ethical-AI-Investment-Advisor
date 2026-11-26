"""
Fireworks AI LLM - Real Language Model with Automatic Three-Tier Failover
Prime Model: accounts/fireworks/models/deepseek-v4p1-flash (High accuracy, ultra-fast, low cost)
Fallback Model: accounts/fireworks/models/glm-5p3-flash (Open-weights model, served by Fireworks)
Tertiary Fallback: local open-weights Hugging Face model (no API key; see app/services/local_llm.py)

Without FIREWORKS_API_KEY, get_llm() returns LocalLLM, which uses only the local model.
No tier ever returns a canned portfolio: if every model fails, the answer is an error.
"""
import requests
import os
import json
import re
import hashlib
import time
from typing import Dict, List, Optional, Tuple
from app.core.config import FIREWORKS_API_KEY, FIREWORKS_MODEL, FIREWORKS_FALLBACK_MODEL
from app.schemas.schemas import PortfolioRecommendation
from app.services.local_llm import LocalHFModel

DEFAULT_PRIME_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"
DEFAULT_FALLBACK_MODEL = "accounts/fireworks/models/glm-5p3-flash"

class _RecommendationLLM:
    """
    Shared prompt building + strict parsing. Subclasses implement
    generate_with_meta(prompt, max_tokens, temperature) -> (text, model_tier, model_name, failure_type);
    model_tier "unavailable" means no model produced an answer.
    """

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2
    ) -> str:
        """
        Generate text with automatic failover across the configured model tiers
        """
        text, tier, _, failure = self.generate_with_meta(prompt, max_tokens=max_tokens, temperature=temperature)
        if tier == "unavailable":
            raise RuntimeError(f"No LLM tier produced an answer ({failure})")
        return text

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

        t0 = time.time()
        response_text, model_tier, model_used, call_failure = self.generate_with_meta(
            prompt=prompt,
            max_tokens=4096,
            temperature=temperature
        )
        llm_latency_s = round(time.time() - t0, 4)

        if model_tier == "unavailable":
            # Every tier failed: there is no LLM answer, so report an error rather than invent one.
            recommendation = {
                "refused": False,
                "recommended_products": [],
                "allocation_percentages": [],
                "rationale": "",
                "parse_error": f"All LLM tiers failed ({call_failure or 'api_error'}); no LLM answer.",
            }
        else:
            recommendation = self._parse_recommendation(response_text, retrieved_products)
        recommendation["prompt_hash"] = prompt_hash
        recommendation["model_name"] = model_used
        recommendation["model_tier"] = model_tier
        recommendation["temperature"] = temperature
        recommendation["llm_latency_s"] = llm_latency_s
        if model_tier == "unavailable":
            recommendation["failure_type"] = call_failure or "api_error"
        elif recommendation.get("parse_error"):
            recommendation["failure_type"] = "broken_json"
        else:
            # An earlier tier may have failed while a later one still answered: record it, but it is a genuine answer
            recommendation["failure_type"] = None
            recommendation["primary_failure"] = call_failure
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
6. If none of the listed products can satisfy the constraints, respond with {{"no_suitable_products": true, "reason": "..."}} instead of a portfolio.

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
            if not isinstance(recommendation, dict):
                raise ValueError("JSON response is not an object")

            # Explicit refusal: the LLM says no listed product fits the constraints
            if recommendation.get('no_suitable_products') is True:
                return {
                    "refused": True,
                    "reason": str(recommendation.get('reason') or "LLM reported no suitable products."),
                    "recommended_products": [],
                    "allocation_percentages": [],
                    "rationale": str(recommendation.get('reason') or ""),
                    "parse_error": None,
                }

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
                "refused": False,
                "parse_error": None,
            }

        except Exception as e:
            print(f"⚠️  Failed to parse/validate LLM response: {e}")
            print(f"   Raw Response: {str(response_text)[:200]}")

            # Never invent a portfolio here: an unparseable answer is an error, not a choice.
            return {
                "refused": False,
                "recommended_products": [],
                "allocation_percentages": [],
                "rationale": "",
                "parse_error": str(e),
                "failure_type": "broken_json",
            }


class FireworksLLM(_RecommendationLLM):
    """
    Real LLM client for Fireworks AI with three-tier failover (prime -> fallback -> local HF model).
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
        self.local_model = LocalHFModel()
        
        print(f"✅ Fireworks AI LLM initialized")
        print(f"   Prime Model: {self.prime_model}")
        print(f"   Fallback Model: {self.fallback_model}")
        print(f"   Local Model: {self.local_model.model_id} ({self.local_model.backend}, loaded on first use)")

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

    def generate_with_meta(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2
    ) -> Tuple[str, str, str, Optional[str]]:
        """Return (response_text, model_tier, model_name, failure_type)."""
        try:
            return self._call_fireworks_api(self.prime_model, prompt, max_tokens, temperature), "primary", self.prime_model, None
        except requests.Timeout as prime_err:
            print(f"⚠️  Fireworks prime model '{self.prime_model}' timeout: {prime_err}")
            prime_fail = "timeout"
        except Exception as prime_err:
            print(f"⚠️  Fireworks prime model '{self.prime_model}' error: {prime_err}")
            prime_fail = "api_error"

        print(f"🔄 Switching to fallback model: '{self.fallback_model}'...")
        try:
            res = self._call_fireworks_api(self.fallback_model, prompt, max_tokens, temperature)
            print(f"✅ Generated output with fallback model '{self.fallback_model}'.")
            return res, "fallback", self.fallback_model, prime_fail
        except requests.Timeout as fallback_err:
            print(f"⚠️  Fireworks fallback model '{self.fallback_model}' timeout: {fallback_err}")
            fb_fail = "timeout"
        except Exception as fallback_err:
            print(f"⚠️  Fireworks fallback model '{self.fallback_model}' failed: {fallback_err}")
            fb_fail = "api_error"

        print(f"🛡️  Falling back to local open-weights model '{self.local_model.model_id}'...")
        try:
            res = self.local_model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            return res, "local", self.local_model.name, fb_fail
        except Exception as local_err:
            print(f"⚠️  Local model failed: {local_err}")
            return "", "unavailable", None, fb_fail



class LocalLLM(_RecommendationLLM):
    """Recommendation LLM backed only by a local Hugging Face open-weights model (no API key)."""

    def __init__(self, model_id: str = None, backend: str = None):
        self.local_model = LocalHFModel(model_id=model_id, backend=backend)
        print(f"✅ Local open-weights LLM initialized: {self.local_model.model_id} ({self.local_model.backend}, loaded on first use)")

    def generate_with_meta(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2
    ) -> Tuple[str, str, str, Optional[str]]:
        try:
            return self.local_model.generate(prompt, max_tokens=max_tokens, temperature=temperature), "local", self.local_model.name, None
        except Exception as local_err:
            print(f"⚠️  Local model failed: {local_err}")
            return "", "unavailable", None, "api_error"


def get_llm(use_real: bool = True) -> _RecommendationLLM:
    """
    Fireworks (with local HF fallback) when an API key is set; otherwise the local HF model alone.
    """
    if use_real and FIREWORKS_API_KEY:
        try:
            return FireworksLLM()
        except Exception as e:
            print(f"⚠️  Could not initialize FireworksLLM: {e}")
            print("⚠️  Falling back to local open-weights model")
            return LocalLLM()
    if use_real:
        print("ℹ️  No FIREWORKS_API_KEY detected. Running with local open-weights model (no API key needed).")
    return LocalLLM()
