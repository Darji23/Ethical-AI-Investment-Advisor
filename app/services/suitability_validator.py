"""
Core Suitability Validation Engine — Config-Driven Regulatory & Firm Policy Rules.

Loads versioned rule definitions and thresholds from config/suitability_rules.json.
Regulatory Framework:
- Governing Standard for Retail Recommendations: SEC Regulation Best Interest (SEA Rule 15l-1, "Reg BI").
  FINRA Rule 2111 does not apply to recommendations subject to Reg BI, but its list of customer
  profile factors remains the industry benchmark for suitability assessment.
- FINRA Regulatory Notice 09-31: Non-Traditional (Leveraged and Inverse) ETFs.
- FINRA Regulatory Notice 12-03: Heightened Supervision of Complex Products.
- Rule 6 (Diversification / Concentration) is plain Firm Policy.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from app.schemas.schemas import (
    InvestorProfile,
    Product,
    PortfolioRecommendation,
    SuitabilityResult,
)

DEFAULT_RULES_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "suitability_rules.json"


@dataclass(frozen=True)
class SuitabilityRule:
    """Declarative definition of a regulatory suitability rule."""
    rule_id: str
    name: str
    regulatory_source: str
    firm_policy_threshold: str
    category: str
    description: str
    short_id: str = ""


def load_rules_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load versioned suitability rules and thresholds from JSON config."""
    path = config_path or DEFAULT_RULES_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_INITIAL_CONFIG = load_rules_config()
RULE_SET_VERSION: str = _INITIAL_CONFIG["rule_set_version"]
SHARED_SUITABILITY_RULES: List[SuitabilityRule] = [
    SuitabilityRule(
        rule_id=r["rule_id"],
        short_id=r.get("short_id", r["rule_id"]),
        name=r["name"],
        regulatory_source=r["regulatory_source"],
        firm_policy_threshold=r["firm_policy_threshold"],
        category=r["category"],
        description=r["description"],
    )
    for r in _INITIAL_CONFIG["rules"]
]


class SuitabilityValidator:
    """
    Validates investment recommendations against versioned regulatory suitability
    rules and firm policies loaded from config/suitability_rules.json.
    """

    def __init__(self, product_master: List[Product], config_path: Optional[Path] = None):
        self.config = load_rules_config(config_path)
        self.rule_set_version: str = self.config["rule_set_version"]
        self.thresholds: Dict[str, float] = self.config["thresholds"]
        self.rules: List[SuitabilityRule] = [
            SuitabilityRule(
                rule_id=r["rule_id"],
                short_id=r.get("short_id", r["rule_id"]),
                name=r["name"],
                regulatory_source=r["regulatory_source"],
                firm_policy_threshold=r["firm_policy_threshold"],
                category=r["category"],
                description=r["description"],
            )
            for r in self.config["rules"]
        ]
        self.products: Dict[str, Product] = {p.ticker: p for p in product_master}
        self.products_by_risk = self._categorize_by_risk()
        self.products_by_asset_class = self._categorize_by_asset_class()

    def _categorize_by_risk(self) -> Dict[int, List[str]]:
        categories = {1: [], 2: [], 3: [], 4: [], 5: []}
        for ticker, product in self.products.items():
            categories[product.risk_score].append(ticker)
        return categories

    def _categorize_by_asset_class(self) -> Dict[str, List[str]]:
        categories: Dict[str, List[str]] = {}
        for ticker, product in self.products.items():
            categories.setdefault(product.asset_class, []).append(ticker)
        return categories

    def get_rule_table(self) -> List[Dict[str, Any]]:
        """Return the shared rule table metadata for audits, documentation, and UI display."""
        return [
            {
                "rule_id": r.rule_id,
                "short_id": r.short_id,
                "name": r.name,
                "regulatory_source": r.regulatory_source,
                "firm_policy_threshold": r.firm_policy_threshold,
                "category": r.category,
                "description": r.description,
                "rule_set_version": self.rule_set_version,
            }
            for r in self.rules
        ]

    # ─────────────────────────────────────────────────────────────────────────
    #  PRODUCT-LEVEL SUITABILITY EVALUATION (Single Source of Truth)
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_product_suitability(
        self,
        profile: InvestorProfile,
        product: Product,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Evaluate a single product against all relevant shared suitability rules.
        Returns (is_suitable, violations, notes).
        """
        violations: List[str] = []
        notes: List[str] = []
        t = self.thresholds

        is_lev = bool(
            getattr(product, "is_leveraged", False)
            or getattr(product, "is_inverse", False)
            or getattr(product, "leverage_factor", 1.0) != 1.0
        )

        # ── R1: Non-Traditional (Leveraged & Inverse) ETF Exclusion ──────────
        if is_lev:
            lev_tag = f"{product.leverage_factor:+.0f}x"
            violations.append(
                f"R1: leveraged/inverse ETF excluded ({product.ticker}, {lev_tag}) — FINRA 09-31 / 12-03."
            )

        # ── R2: Time Horizon, Liquidity & Preservation ───────────────────────
        short_horizon = int(t["short_horizon_years"])
        if profile.time_horizon_years < short_horizon:
            if product.risk_score >= 4:
                violations.append(
                    f"R2: horizon {profile.time_horizon_years}y < {short_horizon}y forbids Risk {product.risk_score} ({product.ticker})."
                )
            if product.asset_class == "alternative":
                violations.append(
                    f"R2: horizon {profile.time_horizon_years}y < {short_horizon}y forbids alternative asset ({product.ticker})."
                )

        if getattr(profile, "liquidity_needs", "medium") == "high":
            liq_cap = int(t["high_liquidity_max_risk"])
            if product.risk_score > liq_cap:
                violations.append(
                    f"R2: high liquidity needs, risk cap {liq_cap} ({product.ticker} is Risk {product.risk_score})."
                )

        if getattr(profile, "investment_objective", "balanced") == "preservation":
            pres_cap = int(t["preservation_max_risk"])
            if product.risk_score > pres_cap:
                violations.append(
                    f"R2: preservation objective, risk cap {pres_cap} ({product.ticker} is Risk {product.risk_score})."
                )

        # ── R3: Customer Risk Tolerance Alignment ────────────────────────────
        max_offset = int(t["max_product_risk_offset"])
        if product.risk_score > profile.risk_tolerance + max_offset:
            violations.append(
                f"R3: product risk {product.risk_score} exceeds tolerance {profile.risk_tolerance} + {max_offset} ({product.ticker})."
            )
        elif product.risk_score == profile.risk_tolerance + max_offset:
            notes.append(
                f"R3 Disclosure: {product.ticker} is 1 risk level above customer tolerance ({profile.risk_tolerance})."
            )

        # ── R4: Senior Investor Protection (Age Rules) ───────────────────────
        cap_70 = int(t["senior_70_max_risk"])
        cap_65 = int(t["senior_65_max_risk"])
        if profile.age >= 70:
            if product.risk_score > cap_70:
                violations.append(
                    f"R4: age 70+, risk cap {cap_70} ({product.ticker} is Risk {product.risk_score})."
                )
        elif profile.age >= 65:
            if product.risk_score > cap_65:
                violations.append(
                    f"R4: age 65-69, risk cap {cap_65} ({product.ticker} is Risk {product.risk_score})."
                )
        elif profile.age >= 60:
            if product.risk_score == int(t["senior_60_disclosure_risk"]):
                notes.append(
                    f"R4 Disclosure: Risk 5 product {product.ticker} for age {profile.age} requires risk acknowledgment."
                )

        # ── R5: Investor Experience & Complex Products ───────────────────────
        beg_cap = int(t["beginner_max_risk"])
        if profile.investment_experience == "beginner":
            if product.risk_score > beg_cap:
                violations.append(
                    f"R5: beginner investor, risk cap {beg_cap} ({product.ticker} is Risk {product.risk_score})."
                )
            if product.asset_class == "alternative":
                violations.append(
                    f"R5: beginner investor cannot hold alternative asset ({product.ticker}) — FINRA 12-03."
                )
            if getattr(product, "is_futures_based", False):
                violations.append(
                    f"R5: beginner investor cannot hold futures-based product ({product.ticker}) — FINRA 12-03."
                )
        elif profile.investment_experience == "intermediate":
            if product.risk_score == 5 and not is_lev:
                notes.append(
                    f"R5 Disclosure: Intermediate investor holding Risk 5 product {product.ticker} requires risk acknowledgment."
                )

        return (len(violations) == 0, violations, notes)

    def get_max_risk_for_investor(self, profile: InvestorProfile) -> int:
        """
        Absolute maximum product risk score allowed for this investor.
        Guaranteed to match every risk-capping rule in evaluate_product_suitability.
        """
        t = self.thresholds
        max_risk = profile.risk_tolerance + int(t["max_product_risk_offset"])

        if profile.time_horizon_years < int(t["short_horizon_years"]):
            max_risk = min(max_risk, 3)

        if getattr(profile, "liquidity_needs", "medium") == "high":
            max_risk = min(max_risk, int(t["high_liquidity_max_risk"]))

        if getattr(profile, "investment_objective", "balanced") == "preservation":
            max_risk = min(max_risk, int(t["preservation_max_risk"]))

        if profile.age >= 70:
            max_risk = min(max_risk, int(t["senior_70_max_risk"]))
        elif profile.age >= 65:
            max_risk = min(max_risk, int(t["senior_65_max_risk"]))

        if profile.investment_experience == "beginner":
            max_risk = min(max_risk, int(t["beginner_max_risk"]))

        return max_risk

    def get_suitable_products(self, profile: InvestorProfile) -> List[str]:
        """
        Return all product tickers guaranteed to pass evaluate_product_suitability(profile, product).
        Consistent with Tier 1 pre-filtering.
        """
        suitable = []
        for ticker, product in self.products.items():
            is_suitable, _, _ = self.evaluate_product_suitability(profile, product)
            if is_suitable:
                suitable.append(ticker)
        return suitable

    # ─────────────────────────────────────────────────────────────────────────
    #  REBALANCING & SAFE FALLBACK HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def rebalance_allocations(
        recommendation: PortfolioRecommendation,
        approved_tickers: List[str],
    ) -> Dict[str, float]:
        """
        Proportionally redistribute weight across approved products so allocations sum to 100.0%.
        Returns {ticker: pct} or {} if no approved products remain.
        """
        if not approved_tickers:
            return {}

        raw_weights: Dict[str, float] = {}
        for ticker in approved_tickers:
            if ticker in recommendation.recommended_products:
                idx = recommendation.recommended_products.index(ticker)
                raw_weights[ticker] = raw_weights.get(ticker, 0.0) + float(
                    recommendation.allocation_percentages[idx]
                )

        total = sum(raw_weights.values())
        if total <= 0:
            equal = round(100.0 / len(approved_tickers), 2)
            rebalanced = {t: equal for t in approved_tickers}
        else:
            rebalanced = {t: round((w / total) * 100.0, 2) for t, w in raw_weights.items()}

        diff = round(100.0 - sum(rebalanced.values()), 2)
        if diff != 0.0 and rebalanced:
            largest = max(rebalanced, key=rebalanced.get)
            rebalanced[largest] = round(rebalanced[largest] + diff, 2)

        return rebalanced

    def build_safe_fallback_portfolio(
        self,
        profile: InvestorProfile,
        reason: str = "Fallback rule-based portfolio",
    ) -> Optional[PortfolioRecommendation]:
        """
        Deterministic, guaranteed-compliant rule-based portfolio for the investor's profile.
        Always uses at least two suitable products, each at <= 60% allocation.
        Returns None if fewer than 2 suitable products exist (caller must refer to a human advisor).
        """
        suitable_tickers = self.get_suitable_products(profile)
        suitable_prods = [self.products[t] for t in suitable_tickers]

        if len(suitable_prods) < 2:
            return None

        safe_prods = sorted(
            [p for p in suitable_prods if p.asset_class in ("cash", "fixed_income")],
            key=lambda p: (p.risk_score, p.expense_ratio_pct),
        )
        core_prods = sorted(
            [p for p in suitable_prods if p.asset_class == "equity" and p.risk_score <= 3],
            key=lambda p: (abs(p.risk_score - min(3, profile.risk_tolerance)), p.expense_ratio_pct),
        )

        # Prefer defensive + core equity when horizon/liquidity allow; otherwise 2 defensive/lowest-risk products
        if safe_prods and core_prods and profile.risk_tolerance >= 3 and profile.time_horizon_years >= 3:
            if getattr(profile, "liquidity_needs", "medium") == "high" or getattr(profile, "investment_objective", "") == "preservation":
                chosen = [safe_prods[0].ticker, core_prods[0].ticker]
                allocs = [60.0, 40.0]
            else:
                chosen = [core_prods[0].ticker, safe_prods[0].ticker]
                allocs = [60.0, 40.0]
        elif len(safe_prods) >= 2:
            chosen = [safe_prods[0].ticker, safe_prods[1].ticker]
            allocs = [60.0, 40.0]
        else:
            ordered = sorted(suitable_prods, key=lambda p: p.risk_score)
            chosen = [ordered[0].ticker, ordered[1].ticker]
            allocs = [60.0, 40.0]

        rec = PortfolioRecommendation(
            recommended_products=chosen,
            allocation_percentages=allocs,
            rationale=f"Deterministic rule-based safe fallback ({reason}).",
        )
        # Verify compliance against validator before returning
        check = self.validate_recommendation(profile, rec)
        if not check.is_suitable:
            if len(safe_prods) >= 2:
                rec = PortfolioRecommendation(
                    recommended_products=[safe_prods[0].ticker, safe_prods[1].ticker],
                    allocation_percentages=[50.0, 50.0],
                    rationale=f"Deterministic defensive safe fallback ({reason}).",
                )
                if self.validate_recommendation(profile, rec).is_suitable:
                    return rec
            return None
        return rec

    # ─────────────────────────────────────────────────────────────────────────
    #  PORTFOLIO-LEVEL SUITABILITY VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    def validate_recommendation(
        self,
        profile: InvestorProfile,
        recommendation: PortfolioRecommendation,
    ) -> SuitabilityResult:
        """
        Validate recommended portfolio against product-level and portfolio-level rules.
        Proportionally rebalances approved products if any products were rejected.
        """
        violations: List[str] = []
        compliance_notes: List[str] = []
        approved: List[str] = []
        rejected: List[str] = []
        t = self.thresholds

        for w in getattr(profile, "profile_warnings", []) or []:
            compliance_notes.append(f"Profile Warning: {w}")

        # 1. Product-Level Checks
        for ticker in recommendation.recommended_products:
            if ticker not in self.products:
                violations.append(
                    f"UNKNOWN_TICKER: Product '{ticker}' is not in the approved catalog."
                )
                rejected.append(ticker)
            else:
                product = self.products[ticker]
                is_ok, p_viols, p_notes = self.evaluate_product_suitability(profile, product)
                if is_ok:
                    approved.append(ticker)
                    compliance_notes.extend(p_notes)
                else:
                    rejected.append(ticker)
                    violations.extend(p_viols)
                    compliance_notes.extend(p_notes)

        rebalanced_map: Optional[Dict[str, float]] = None
        effective_allocs: Dict[str, float] = {}
        if approved:
            if rejected:
                rebalanced_map = self.rebalance_allocations(recommendation, approved)
                effective_allocs = rebalanced_map
                compliance_notes.append(
                    f"Rebalanced {len(approved)} approved products to 100.0% after removing {rejected}."
                )
            else:
                for ticker in approved:
                    idx = recommendation.recommended_products.index(ticker)
                    effective_allocs[ticker] = effective_allocs.get(ticker, 0.0) + float(
                        recommendation.allocation_percentages[idx]
                    )

        # 2. Portfolio-Level Constraints
        if approved:
            asset_class_alloc: Dict[str, float] = {}
            high_risk_alloc = 0.0

            for ticker in approved:
                alloc = effective_allocs[ticker]
                product = self.products[ticker]

                # R6: Single Product Concentration (warning > 50%, block > 60%)
                warn_pct = t["single_product_warning_pct"]
                max_pct = t["single_product_max_pct"]
                if alloc > max_pct:
                    violations.append(
                        f"R6: {ticker} allocation ({alloc:.1f}%) exceeds {max_pct:.0f}% single-product cap."
                    )
                elif alloc > warn_pct:
                    compliance_notes.append(
                        f"R6 Warning: {ticker} allocation ({alloc:.1f}%) exceeds {warn_pct:.0f}% concentration threshold."
                    )

                ac = product.asset_class
                asset_class_alloc[ac] = asset_class_alloc.get(ac, 0.0) + alloc
                if product.risk_score >= 4:
                    high_risk_alloc += alloc

            ac_warn = t["asset_class_warning_pct"]
            for ac, alloc in asset_class_alloc.items():
                if alloc > ac_warn:
                    compliance_notes.append(
                        f"R6 Warning: Portfolio is {alloc:.1f}% concentrated in {ac} (> {ac_warn:.0f}%)."
                    )

            safe_alloc = asset_class_alloc.get("fixed_income", 0.0) + asset_class_alloc.get("cash", 0.0)
            if profile.time_horizon_years < int(t["short_horizon_years"]):
                min_safe = t["short_horizon_min_safe_pct"]
                if safe_alloc < min_safe:
                    violations.append(
                        f"R2: short horizon ({profile.time_horizon_years} yrs) requires >= {min_safe:.0f}% fixed income/cash (got {safe_alloc:.1f}%)."
                    )
                else:
                    compliance_notes.append(
                        f"R2 Policy Met: {safe_alloc:.1f}% in fixed income/cash for short horizon."
                    )

            if getattr(profile, "liquidity_needs", "medium") == "high":
                min_liq_safe = t["high_liquidity_min_safe_pct"]
                if safe_alloc < min_liq_safe:
                    violations.append(
                        f"R2: high liquidity needs require >= {min_liq_safe:.0f}% fixed income/cash (got {safe_alloc:.1f}%)."
                    )

            investable = getattr(profile, "investable_assets", profile.net_worth)
            max_hr_pct = (
                t["low_assets_max_high_risk_pct"]
                if investable < t["low_assets_threshold_usd"]
                else t["general_max_high_risk_pct"]
            )
            if high_risk_alloc > max_hr_pct:
                violations.append(
                    f"R7: high-risk (Risk 4-5) allocation ({high_risk_alloc:.1f}%) exceeds {max_hr_pct:.0f}% cap for investable_assets=${investable:,.0f}."
                )

        # 3. R3: Weighted Average Portfolio Risk Cap
        portfolio_risk = self._calculate_portfolio_risk(effective_allocs, approved)
        max_port_risk = profile.risk_tolerance + t["max_portfolio_risk_offset"]
        if approved and portfolio_risk > max_port_risk:
            violations.append(
                f"R3: portfolio weighted risk ({portfolio_risk:.2f}) exceeds cap ({max_port_risk:.2f})."
            )

        is_suitable = len(violations) == 0 and len(approved) > 0

        if is_suitable:
            compliance_notes.append(
                f"Portfolio approved: {len(approved)} products within suitability rules (rule_set={self.rule_set_version})."
            )

        return SuitabilityResult(
            is_suitable=is_suitable,
            risk_violations=violations,
            compliance_notes=compliance_notes,
            approved_products=approved,
            rejected_products=rejected,
            overall_portfolio_risk=round(portfolio_risk, 2),
            rebalanced_allocations=rebalanced_map,
            audit_metadata={"rule_set_version": self.rule_set_version},
        )

    def _calculate_portfolio_risk(
        self,
        allocs_by_ticker: Dict[str, float],
        approved_tickers: List[str],
    ) -> float:
        if not approved_tickers:
            return 0.0
        total_weight = 0.0
        weighted_risk = 0.0
        for ticker in approved_tickers:
            if ticker in allocs_by_ticker:
                weight = allocs_by_ticker[ticker] / 100.0
                weighted_risk += self.products[ticker].risk_score * weight
                total_weight += weight
        return (weighted_risk / total_weight) if total_weight > 0 else 0.0
