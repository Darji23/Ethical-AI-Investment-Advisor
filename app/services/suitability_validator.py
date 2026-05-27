"""
Core Suitability Validation Engine (Updated for Expanded Dataset)
Validates LLM portfolio recommendations against investor profiles
"""
from typing import List, Dict
from app.schemas.schemas import (
    InvestorProfile, 
    Product, 
    PortfolioRecommendation, 
    SuitabilityResult
)

class SuitabilityValidator:
    """
    Validates investment recommendations against regulatory suitability rules
    """
    
    def __init__(self, product_master: List[Product]):
        """Initialize with approved product list"""
        self.products = {p.ticker: p for p in product_master}
        self.products_by_risk = self._categorize_by_risk()
        self.products_by_asset_class = self._categorize_by_asset_class()
    
    def _categorize_by_risk(self) -> Dict:
        """Group products by risk level for quick lookup"""
        categories = {1: [], 2: [], 3: [], 4: [], 5: []}
        for ticker, product in self.products.items():
            categories[product.risk_score].append(ticker)
        return categories
    
    def _categorize_by_asset_class(self) -> Dict:
        """Group products by asset class"""
        categories = {}
        for ticker, product in self.products.items():
            if product.asset_class not in categories:
                categories[product.asset_class] = []
            categories[product.asset_class].append(ticker)
        return categories
    
    def get_max_risk_for_investor(self, profile: InvestorProfile) -> int:
        """Get the absolute maximum product risk score allowed for this investor"""
        max_risk = profile.risk_tolerance + 1
        
        # Age restrictions
        if profile.age >= 70:
            max_risk = min(max_risk, 3)
        elif profile.age >= 65:
            max_risk = min(max_risk, 4)
            
        # Experience restrictions
        if profile.investment_experience == "beginner":
            max_risk = min(max_risk, 4)
            
        return max_risk
    
    def validate_recommendation(
        self, 
        profile: InvestorProfile, 
        recommendation: PortfolioRecommendation
    ) -> SuitabilityResult:
        """
        Main validation method - checks if recommendation is suitable
        """
        violations = []
        compliance_notes = []
        approved = []
        rejected = []
        
        # Check 1: All products exist in approved list
        for ticker in recommendation.recommended_products:
            if ticker not in self.products:
                violations.append(f"Product {ticker} not in approved list")
                rejected.append(ticker)
            else:
                approved.append(ticker)
        
        # Check 2: Risk alignment
        risk_check = self._validate_risk_alignment(
            profile, 
            recommendation, 
            approved
        )
        violations.extend(risk_check['violations'])
        compliance_notes.extend(risk_check['notes'])
        
        # If risk violations, move products to rejected
        if risk_check['violations']:
            # Track which ones were rejected
            rejected_due_to_risk = [
                ticker for ticker in approved 
                if self.products[ticker].risk_score > profile.risk_tolerance + 1
            ]
            
            # Age-based checks (strict rejections)
            if profile.age >= 70:
                rejected_due_to_risk.extend([
                    ticker for ticker in approved
                    if self.products[ticker].risk_score >= 4
                ])
            elif profile.age >= 65:
                rejected_due_to_risk.extend([
                    ticker for ticker in approved
                    if self.products[ticker].risk_score == 5
                ])
                
            # Experience rejections
            if profile.investment_experience == "beginner":
                rejected_due_to_risk.extend([
                    ticker for ticker in approved
                    if self.products[ticker].risk_score >= 5 or
                       "leveraged" in self.products[ticker].name.lower() or
                       "3x" in self.products[ticker].name
                ])
            
            rejected_set = set(rejected_due_to_risk)
            rejected.extend(list(rejected_set))
            approved = [ticker for ticker in approved if ticker not in rejected_set]
        
        # Check 3: Minimum investment requirements
        investment_check = self._validate_minimum_investment(
            profile,
            recommendation,
            approved
        )
        violations.extend(investment_check['violations'])
        
        # Check 4: Portfolio diversification
        diversification_check = self._validate_diversification(
            recommendation,
            approved
        )
        compliance_notes.extend(diversification_check['notes'])
        if diversification_check['violations']:
            violations.extend(diversification_check['violations'])
        
        # Check 5: Experience-based restrictions
        experience_check = self._validate_experience_alignment(
            profile,
            approved
        )
        violations.extend(experience_check['violations'])
        compliance_notes.extend(experience_check['notes'])
        
        # Calculate overall portfolio risk
        portfolio_risk = self._calculate_portfolio_risk(
            recommendation,
            approved
        )
        
        # Final determination
        is_suitable = len(violations) == 0 and len(approved) > 0
        
        if is_suitable:
            compliance_notes.append(
                f"Portfolio approved: {len(approved)} products within risk tolerance"
            )
            compliance_notes.append(
                f"Portfolio risk score: {portfolio_risk:.2f}/5.0"
            )
        
        return SuitabilityResult(
            is_suitable=is_suitable,
            risk_violations=violations,
            compliance_notes=compliance_notes,
            approved_products=approved,
            rejected_products=rejected,
            overall_portfolio_risk=portfolio_risk
        )
    
    def _validate_risk_alignment(
        self, 
        profile: InvestorProfile, 
        recommendation: PortfolioRecommendation,
        approved_tickers: List[str]
    ) -> Dict:
        """Check if products match investor's risk tolerance"""
        violations = []
        notes = []
        
        for ticker in approved_tickers:
            product = self.products[ticker]
            
            # Hard rule: Product risk cannot exceed investor tolerance by more than 1
            if product.risk_score > profile.risk_tolerance + 1:
                violations.append(
                    f"{ticker} (Risk {product.risk_score}) exceeds "
                    f"tolerance {profile.risk_tolerance} by >1 level"
                )
            
            # Soft warning: Flag if exactly 1 level above
            elif product.risk_score == profile.risk_tolerance + 1:
                notes.append(
                    f"{ticker} is 1 risk level above tolerance - requires disclosure"
                )
        
        # Age-based checks (Enhanced for more granular control)
        if profile.age >= 70:
            # Strict rules for 70+ (no high-risk products)
            high_risk_products = [
                ticker for ticker in approved_tickers
                if self.products[ticker].risk_score >= 4
            ]
            if high_risk_products:
                violations.append(
                    f"High-risk products {high_risk_products} not suitable for age 70+"
                )
        
        elif profile.age >= 65:
            # Moderate restrictions for 65-69 (limit aggressive products)
            aggressive_products = [
                ticker for ticker in approved_tickers
                if self.products[ticker].risk_score == 5
            ]
            if aggressive_products:
                violations.append(
                    f"Aggressive products {aggressive_products} require special approval for age 65-69"
                )
        
        elif profile.age >= 60:
            # Soft warning for 60-64
            aggressive_products = [
                ticker for ticker in approved_tickers
                if self.products[ticker].risk_score == 5
            ]
            if aggressive_products:
                notes.append(
                    f"Warning: Aggressive products {aggressive_products} - ensure investor understands risks"
                )
        
        return {'violations': violations, 'notes': notes}
    
    def _validate_minimum_investment(
        self,
        profile: InvestorProfile,
        recommendation: PortfolioRecommendation,
        approved_tickers: List[str]
    ) -> Dict:
        """Check if investor can meet minimum investment requirements"""
        violations = []
        
        # Assume a hypothetical portfolio size (10% of net worth)
        available_capital = profile.net_worth * 0.10
        
        for i, ticker in enumerate(approved_tickers):
            if ticker in recommendation.recommended_products:
                idx = recommendation.recommended_products.index(ticker)
                allocation_pct = recommendation.allocation_percentages[idx]
                
                product = self.products[ticker]
                required_investment = (allocation_pct / 100.0) * available_capital
                
                if required_investment < product.min_investment:
                    violations.append(
                        f"{ticker} requires ${product.min_investment:,.0f} minimum, "
                        f"but allocation is only ${required_investment:,.0f}"
                    )
        
        return {'violations': violations}
    
    def _validate_diversification(
        self,
        recommendation: PortfolioRecommendation,
        approved_tickers: List[str]
    ) -> Dict:
        """
        Check portfolio diversification
        """
        violations = []
        notes = []
        
        if not approved_tickers:
            return {'violations': violations, 'notes': notes}
        
        # Check 1: Single product concentration (no more than 60%)
        for ticker in approved_tickers:
            if ticker in recommendation.recommended_products:
                idx = recommendation.recommended_products.index(ticker)
                allocation = recommendation.allocation_percentages[idx]
                
                if allocation > 60.0:
                    violations.append(
                        f"{ticker} allocation ({allocation}%) exceeds 60% concentration limit"
                    )
                elif allocation > 50.0:
                    notes.append(
                        f"Warning: {ticker} allocation ({allocation}%) is highly concentrated"
                    )
        
        # Check 2: Asset class diversification
        asset_class_allocation = {}
        for ticker in approved_tickers:
            if ticker in recommendation.recommended_products:
                idx = recommendation.recommended_products.index(ticker)
                allocation = recommendation.allocation_percentages[idx]
                
                asset_class = self.products[ticker].asset_class
                if asset_class not in asset_class_allocation:
                    asset_class_allocation[asset_class] = 0
                asset_class_allocation[asset_class] += allocation
        
        # Flag if single asset class > 80%
        for asset_class, allocation in asset_class_allocation.items():
            if allocation > 80.0:
                notes.append(
                    f"Portfolio is {allocation:.1f}% concentrated in {asset_class} - consider diversification"
                )
        
        # Check 3: Minimum diversification (at least 2 products for portfolios)
        if len(approved_tickers) == 1 and len(recommendation.recommended_products) > 1:
            notes.append(
                "Warning: Portfolio reduced to single product after validation"
            )
        
        return {'violations': violations, 'notes': notes}
    
    def _validate_experience_alignment(
        self,
        profile: InvestorProfile,
        approved_tickers: List[str]
    ) -> Dict:
        """
        Check if products match investor's experience level
        """
        violations = []
        notes = []
        
        if profile.investment_experience == "beginner":
            risky_products = [
                ticker for ticker in approved_tickers
                if self.products[ticker].risk_score >= 5 or
                   "leveraged" in self.products[ticker].name.lower() or
                   "3x" in self.products[ticker].name
            ]
            
            if risky_products:
                violations.append(
                    f"Beginner investors cannot access high-risk/leveraged products: {risky_products}"
                )
            
            moderate_aggressive = [
                ticker for ticker in approved_tickers
                if self.products[ticker].risk_score == 4
            ]
            
            if moderate_aggressive and len(moderate_aggressive) > 1:
                notes.append(
                    f"Beginner investor: Multiple moderate-aggressive products {moderate_aggressive} may require additional education"
                )
        
        elif profile.investment_experience == "intermediate":
            aggressive_products = [
                ticker for ticker in approved_tickers
                if self.products[ticker].risk_score >= 5
            ]
            
            if aggressive_products:
                notes.append(
                    f"Intermediate investor: Aggressive products {aggressive_products} require risk acknowledgment"
                )
        
        return {'violations': violations, 'notes': notes}
    
    def _calculate_portfolio_risk(
        self,
        recommendation: PortfolioRecommendation,
        approved_tickers: List[str]
    ) -> float:
        """Calculate weighted average portfolio risk"""
        if not approved_tickers:
            return 0.0
        
        total_weight = 0.0
        weighted_risk = 0.0
        
        for ticker in approved_tickers:
            if ticker in recommendation.recommended_products:
                idx = recommendation.recommended_products.index(ticker)
                weight = recommendation.allocation_percentages[idx] / 100.0
                
                product = self.products[ticker]
                weighted_risk += product.risk_score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_risk / total_weight
    
    def get_suitable_products(self, profile: InvestorProfile) -> List[str]:
        """
        Get list of all suitable products for an investor profile
        """
        suitable = []
        
        for ticker, product in self.products.items():
            if product.risk_score > profile.risk_tolerance + 1:
                continue
            
            # Age restrictions
            if profile.age >= 70 and product.risk_score >= 4:
                continue
            
            if profile.age >= 65 and product.risk_score == 5:
                continue
            
            # Experience restrictions
            if profile.investment_experience == "beginner":
                if product.risk_score >= 5:
                    continue
                if "leveraged" in product.name.lower() or "3x" in product.name:
                    continue
            
            suitable.append(ticker)
        
        return suitable
