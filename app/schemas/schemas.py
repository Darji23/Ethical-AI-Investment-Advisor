"""
Data models for the Ethical AI Suitability Engine — schema v2
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ConfigDict
from typing import Any, Dict, List, Literal, Optional
from enum import IntEnum


class RiskLevel(IntEnum):
    """Risk levels from 1 (Conservative) to 5 (Aggressive)"""
    CONSERVATIVE = 1
    MODERATE_CONSERVATIVE = 2
    MODERATE = 3
    MODERATE_AGGRESSIVE = 4
    AGGRESSIVE = 5


class InvestorProfile(BaseModel):
    """
    Client investor profile — schema v2.

    Design principle: the schema validates data integrity only (types, ranges,
    allowed values). Suitability logic (e.g. age-risk alignment) lives exclusively
    in SuitabilityValidator, not here. This lets valid edge-case profiles — such as
    a 75-year-old who genuinely self-identifies as aggressive — exist in the system
    so the validator can reason about them properly.
    """
    model_config = ConfigDict(use_enum_values=True, extra="ignore")

    investor_id: str = Field(..., description="Unique identifier, format INV00001")
    age: int = Field(..., ge=18, le=100, description="Age 18–100")
    risk_tolerance: int = Field(..., ge=1, le=5, description="Self-reported risk tolerance 1–5")
    annual_income: float = Field(..., ge=0, description="Gross annual income in USD")
    net_worth: float = Field(..., ge=0, description="Total net worth in USD (may include illiquid assets)")
    investable_assets: float = Field(..., ge=0, description="Liquid/investable assets in USD")
    investment_experience: Literal["beginner", "intermediate", "advanced"] = Field(...)
    time_horizon_years: int = Field(..., ge=1, le=50, description="Investment horizon in years")
    investment_objective: Literal["preservation", "income", "balanced", "growth", "speculation"] = Field(
        ..., description="Primary investment objective"
    )
    liquidity_needs: Literal["low", "medium", "high"] = Field(
        ..., description="How soon investor may need funds back"
    )

    # Optional enrichment fields (populated by generator or advisor)
    segment: Optional[str] = Field(default=None, description="Segment tag, e.g. 'edge_case'")
    profile_warnings: List[str] = Field(default_factory=list, description="Contradictions flagged at generation time")

    # Provenance (set by generator)
    _generator: Optional[str] = None
    _seed: Optional[int] = None

    @model_validator(mode="after")
    def investable_assets_le_net_worth(self):
        if self.investable_assets > self.net_worth:
            raise ValueError(
                f"investable_assets ({self.investable_assets:,.0f}) cannot exceed "
                f"net_worth ({self.net_worth:,.0f})"
            )
        return self


class Product(BaseModel):
    """Investment product with risk classification and market metrics"""
    model_config = ConfigDict(use_enum_values=True, extra="ignore")

    ticker: str = Field(..., description="Product ticker symbol")
    name: str = Field(..., description="Product name")
    risk_score: RiskLevel = Field(..., description="Product risk 1-5")
    asset_class: Literal["equity", "fixed_income", "alternative", "cash", "real_estate", "commodity"] = Field(
        ..., 
        description="Asset class category"
    )
    share_price: float = Field(default=0.0, ge=0.0, description="Current single-share price in USD")
    price_as_of: str = Field(default="", description="Date when share price was recorded (YYYY-MM-DD)")
    description: str = Field(default="", description="Fund objective and mandate description")
    expense_ratio_pct: float = Field(default=0.0, ge=0.0, le=3.0, description="Net expense ratio in percent (0-3%)")
    volatility_pct: float = Field(default=0.01, gt=0.0, lt=200.0, description="Annualized historical volatility in percent (0-200%)")
    max_drawdown_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Historical maximum drawdown in percent (0-100%)")
    fund_family: str = Field(default="", description="Fund issuer/family")
    
    # Leverage & complexity detection fields
    is_leveraged: bool = Field(default=False, description="True if fund uses leverage (>1x or <-1x)")
    is_inverse: bool = Field(default=False, description="True if fund provides inverse/short exposure")
    leverage_factor: float = Field(default=1.0, description="Effective leverage multiplier (e.g. 1.0, 2.0, 3.0, -1.0, -2.0, -3.0)")
    is_futures_based: bool = Field(default=False, description="True if fund tracks futures contracts rather than physical assets")
    
    # Track record, regulatory SRRI & catalog tracking
    history_years: float = Field(default=0.0, ge=0.0, description="Years of available daily price history")
    srri: Optional[int] = Field(default=None, ge=1, le=7, description="Official EU SRRI level (1-7) derived from weekly returns")
    catalog_version: str = Field(default="catalog_v2", description="Catalog release version tag")
    active: bool = Field(default=True, description="Whether product is active in current catalog")
    
    # Backward-compatibility aliases (optional)
    min_investment: Optional[float] = None
    expense_ratio: Optional[float] = None
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None

    @field_validator('share_price', mode='before')
    @classmethod
    def populate_share_price(cls, v, info):
        values = info.data
        if (v is None or v == 0.0) and values.get('min_investment'):
            return float(values['min_investment'])
        return float(v) if v is not None else 0.0

    @field_validator('min_investment', mode='before')
    @classmethod
    def sync_min_investment(cls, v, info):
        if v is not None:
            return float(v)
        return float(info.data.get('share_price', 0.0))

    @field_validator('volatility_pct', mode='before')
    @classmethod
    def populate_volatility(cls, v, info):
        values = info.data
        if (v is None or v == 0.0 or v == 0.01) and values.get('volatility'):
            return float(values['volatility'])
        return max(0.01, min(199.99, float(v))) if v is not None else 0.01

    @field_validator('max_drawdown_pct', mode='before')
    @classmethod
    def populate_drawdown(cls, v, info):
        values = info.data
        if (v is None or v == 0.0) and values.get('max_drawdown'):
            return float(values['max_drawdown'])
        return max(0.0, min(100.0, float(v))) if v is not None else 0.0

    @field_validator('expense_ratio_pct', mode='before')
    @classmethod
    def populate_expense(cls, v, info):
        values = info.data
        if (v is None or v == 0.0) and values.get('expense_ratio'):
            val = float(values['expense_ratio'])
            return val * 100.0 if val < 0.05 else val
        return max(0.0, min(3.0, float(v))) if v is not None else 0.0


class PortfolioRecommendation(BaseModel):
    """LLM-generated portfolio recommendation"""
    model_config = ConfigDict(extra="ignore")

    recommended_products: List[str] = Field(..., description="List of tickers")
    allocation_percentages: List[float] = Field(..., description="% allocation per product")
    rationale: str = Field(..., description="LLM's reasoning")

    @field_validator('allocation_percentages')
    @classmethod
    def validate_allocations_sum_to_100(cls, v):
        if not v:
            return v
        if any(a < 0 for a in v):
            raise ValueError("Allocation percentages cannot be negative")
        total = sum(v)
        if not (99.5 <= total <= 100.5):
            raise ValueError(f"Allocations must sum to 100% (±0.5%), got {total}%")
        return v

    @model_validator(mode="after")
    def validate_allocation_count(self):
        if len(self.allocation_percentages) != len(self.recommended_products):
            raise ValueError("Number of allocations must match number of products")
        return self


class SuitabilityResult(BaseModel):
    """Result of suitability validation"""
    is_suitable: bool
    risk_violations: List[str] = []
    compliance_notes: List[str] = []
    approved_products: List[str] = []
    rejected_products: List[str] = []
    overall_portfolio_risk: float = Field(..., ge=0.0, le=5.0, description="0.0 if all products rejected")
    rebalanced_allocations: Optional[Dict[str, float]] = None
    fallback_used: bool = False
    audit_metadata: Dict[str, Any] = Field(default_factory=dict)
