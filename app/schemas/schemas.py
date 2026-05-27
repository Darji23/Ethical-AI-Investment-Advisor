"""
Data models for the Ethical AI Suitability Engine
"""
from pydantic import BaseModel, Field, validator
from typing import List, Literal
from enum import IntEnum

class RiskLevel(IntEnum):
    """Risk levels from 1 (Conservative) to 5 (Aggressive)"""
    CONSERVATIVE = 1
    MODERATE_CONSERVATIVE = 2
    MODERATE = 3
    MODERATE_AGGRESSIVE = 4
    AGGRESSIVE = 5

class InvestorProfile(BaseModel):
    """Client investor profile with validation rules"""
    investor_id: str = Field(..., description="Unique identifier")
    age: int = Field(..., ge=18, le=100, description="Age must be 18-100")
    risk_tolerance: RiskLevel = Field(..., description="Risk tolerance 1-5")
    annual_income: float = Field(..., ge=0, description="Annual income in USD")
    net_worth: float = Field(..., ge=0, description="Net worth in USD")
    investment_experience: Literal["beginner", "intermediate", "advanced"] = Field(...)
    time_horizon_years: int = Field(..., ge=1, le=50, description="Investment horizon")
    
    @validator('risk_tolerance')
    def validate_risk_for_age(cls, v, values):
        """Age-based risk validation: older investors shouldn't have max risk"""
        if 'age' in values:
            age = values['age']
            if age >= 65 and v >= RiskLevel.AGGRESSIVE:
                raise ValueError(
                    f"Risk level {v} may be unsuitable for age {age} (65+)"
                )
        return v
    
    class Config:
        use_enum_values = True

class Product(BaseModel):
    """Investment product with risk classification"""
    ticker: str = Field(..., description="Product ticker symbol")
    name: str = Field(..., description="Product name")
    risk_score: RiskLevel = Field(..., description="Product risk 1-5")
    asset_class: Literal["equity", "fixed_income", "alternative", "cash", "real_estate"] = Field(
        ..., 
        description="Asset class category"
    )
    min_investment: float = Field(..., ge=0, description="Minimum investment")
    
    class Config:
        use_enum_values = True

class PortfolioRecommendation(BaseModel):
    """LLM-generated portfolio recommendation"""
    recommended_products: List[str] = Field(..., description="List of tickers")
    allocation_percentages: List[float] = Field(..., description="% allocation per product")
    rationale: str = Field(..., description="LLM's reasoning")
    
    @validator('allocation_percentages')
    def validate_allocations_sum_to_100(cls, v):
        """Ensure allocations sum to approximately 100%"""
        total = sum(v)
        if not (99.0 <= total <= 101.0):  # Allow small floating point errors
            raise ValueError(f"Allocations must sum to 100%, got {total}%")
        return v
    
    @validator('allocation_percentages')
    def validate_allocation_count(cls, v, values):
        """Ensure allocation list matches product list length"""
        if 'recommended_products' in values:
            if len(v) != len(values['recommended_products']):
                raise ValueError("Number of allocations must match number of products")
        return v

class SuitabilityResult(BaseModel):
    """Result of suitability validation"""
    is_suitable: bool
    risk_violations: List[str] = []
    compliance_notes: List[str] = []
    approved_products: List[str] = []
    rejected_products: List[str] = []
    overall_portfolio_risk: float = Field(..., ge=0.0, le=5.0, description="0.0 if all products rejected")
