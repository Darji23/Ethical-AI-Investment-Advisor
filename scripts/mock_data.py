"""
Generate expanded mock datasets for testing the suitability engine
100 Investor Profiles + 25 Investment Products
Writes files directly to the data/ directory.
"""
import json
import random
import os
from typing import List
from app.schemas.schemas import InvestorProfile, Product, RiskLevel

# Set seed for reproducibility
random.seed(42)

def generate_investor_profiles(count: int = 500) -> List[InvestorProfile]:
    """
    Create diverse investor profiles across different demographics
    """
    profiles = []
    beg_count = int(count * 0.3)
    int_count = int(count * 0.5)
    adv_count = count - beg_count - int_count
    
    experience_levels = (
        ["beginner"] * beg_count + 
        ["intermediate"] * int_count + 
        ["advanced"] * adv_count
    )
    random.shuffle(experience_levels)
    
    for i in range(count):
        investor_id = f"INV{i+1:03d}"  # INV001 to INV100
        age = random.randint(22, 75)
        
        # Risk tolerance based on age (younger = more aggressive)
        if age < 30:
            risk_weights = [0.05, 0.15, 0.30, 0.35, 0.15]
        elif age < 45:
            risk_weights = [0.10, 0.20, 0.40, 0.20, 0.10]
        elif age < 60:
            risk_weights = [0.20, 0.35, 0.30, 0.10, 0.05]
        else:
            risk_weights = [0.40, 0.35, 0.20, 0.05, 0.00]
        
        risk_tolerance = random.choices(
            list(RiskLevel), 
            weights=risk_weights
        )[0]
        
        # Income (correlated with age and experience)
        base_income = random.randint(35000, 150000)
        if experience_levels[i] == "advanced":
            base_income = int(base_income * random.uniform(1.5, 2.5))
        elif age > 45:
            base_income = int(base_income * random.uniform(1.2, 1.8))
        
        annual_income = min(base_income, 500000)  # Cap at $500k
        
        # Net worth (generally increases with age)
        age_multiplier = (age - 20) / 10
        base_net_worth = annual_income * random.uniform(1, 4) * age_multiplier
        net_worth = max(10000, min(base_net_worth, 5000000))  # $10k to $5M
        
        # Time horizon (inversely correlated with age)
        time_horizon_years = max(5, min(40, 75 - age + random.randint(-5, 10)))
        
        # Guard against age >= 65 and risk_tolerance == 5
        if age >= 65 and risk_tolerance == RiskLevel.AGGRESSIVE:
            risk_tolerance = RiskLevel.MODERATE_AGGRESSIVE
            
        profile = InvestorProfile(
            investor_id=investor_id,
            age=age,
            risk_tolerance=risk_tolerance,
            annual_income=round(annual_income, 2),
            net_worth=round(net_worth, 2),
            investment_experience=experience_levels[i],
            time_horizon_years=time_horizon_years
        )
        profiles.append(profile)
    
    return profiles


def generate_product_master() -> List[Product]:
    """
    Create 25 diverse investment products across all risk levels and asset classes
    """
    products = [
        # CONSERVATIVE PRODUCTS (Risk Level 1)
        Product(
            ticker="GOVT",
            name="US Treasury Bond Fund",
            risk_score=RiskLevel.CONSERVATIVE,
            asset_class="fixed_income",
            min_investment=1000.0
        ),
        Product(
            ticker="SHV",
            name="Short-Term Treasury ETF",
            risk_score=RiskLevel.CONSERVATIVE,
            asset_class="fixed_income",
            min_investment=500.0
        ),
        Product(
            ticker="VMFXX",
            name="Money Market Fund",
            risk_score=RiskLevel.CONSERVATIVE,
            asset_class="fixed_income",
            min_investment=3000.0
        ),
        Product(
            ticker="VTIP",
            name="Treasury Inflation-Protected Securities",
            risk_score=RiskLevel.CONSERVATIVE,
            asset_class="fixed_income",
            min_investment=1000.0
        ),
        Product(
            ticker="BND",
            name="Total Bond Market ETF",
            risk_score=RiskLevel.CONSERVATIVE,
            asset_class="fixed_income",
            min_investment=1000.0
        ),
        
        # MODERATE CONSERVATIVE PRODUCTS (Risk Level 2)
        Product(
            ticker="VBTLX",
            name="Vanguard Total Bond Market",
            risk_score=RiskLevel.MODERATE_CONSERVATIVE,
            asset_class="fixed_income",
            min_investment=3000.0
        ),
        Product(
            ticker="AGG",
            name="Core Bond ETF",
            risk_score=RiskLevel.MODERATE_CONSERVATIVE,
            asset_class="fixed_income",
            min_investment=1000.0
        ),
        Product(
            ticker="LQD",
            name="Investment Grade Corporate Bonds",
            risk_score=RiskLevel.MODERATE_CONSERVATIVE,
            asset_class="fixed_income",
            min_investment=2000.0
        ),
        Product(
            ticker="MUB",
            name="Municipal Bond ETF",
            risk_score=RiskLevel.MODERATE_CONSERVATIVE,
            asset_class="fixed_income",
            min_investment=1500.0
        ),
        Product(
            ticker="SCHP",
            name="TIPS ETF",
            risk_score=RiskLevel.MODERATE_CONSERVATIVE,
            asset_class="fixed_income",
            min_investment=1000.0
        ),
        
        # MODERATE PRODUCTS (Risk Level 3)
        Product(
            ticker="VTI",
            name="Vanguard Total Stock Market",
            risk_score=RiskLevel.MODERATE,
            asset_class="equity",
            min_investment=1000.0
        ),
        Product(
            ticker="VWELX",
            name="Vanguard Wellington Fund",
            risk_score=RiskLevel.MODERATE,
            asset_class="equity",
            min_investment=3000.0
        ),
        Product(
            ticker="SPY",
            name="S&P 500 ETF",
            risk_score=RiskLevel.MODERATE,
            asset_class="equity",
            min_investment=500.0
        ),
        Product(
            ticker="VNQ",
            name="Real Estate Investment Trust ETF",
            risk_score=RiskLevel.MODERATE,
            asset_class="real_estate",
            min_investment=2000.0
        ),
        Product(
            ticker="VXUS",
            name="Total International Stock ETF",
            risk_score=RiskLevel.MODERATE,
            asset_class="equity",
            min_investment=1000.0
        ),
        Product(
            ticker="VBIAX",
            name="Balanced Index Fund",
            risk_score=RiskLevel.MODERATE,
            asset_class="equity",
            min_investment=3000.0
        ),
        Product(
            ticker="DIA",
            name="Dow Jones Industrial Average ETF",
            risk_score=RiskLevel.MODERATE,
            asset_class="equity",
            min_investment=1000.0
        ),
        
        # MODERATE AGGRESSIVE PRODUCTS (Risk Level 4)
        Product(
            ticker="ARKK",
            name="ARK Innovation ETF",
            risk_score=RiskLevel.MODERATE_AGGRESSIVE,
            asset_class="equity",
            min_investment=500.0
        ),
        Product(
            ticker="QQQ",
            name="Invesco QQQ Trust",
            risk_score=RiskLevel.MODERATE_AGGRESSIVE,
            asset_class="equity",
            min_investment=1000.0
        ),
        Product(
            ticker="VGT",
            name="Technology Sector ETF",
            risk_score=RiskLevel.MODERATE_AGGRESSIVE,
            asset_class="equity",
            min_investment=1500.0
        ),
        Product(
            ticker="SOXX",
            name="Semiconductor ETF",
            risk_score=RiskLevel.MODERATE_AGGRESSIVE,
            asset_class="equity",
            min_investment=2000.0
        ),
        Product(
            ticker="VOOG",
            name="S&P 500 Growth ETF",
            risk_score=RiskLevel.MODERATE_AGGRESSIVE,
            asset_class="equity",
            min_investment=1000.0
        ),
        
        # AGGRESSIVE PRODUCTS (Risk Level 5)
        Product(
            ticker="BITW",
            name="Crypto Thematic ETF",
            risk_score=RiskLevel.AGGRESSIVE,
            asset_class="alternative",
            min_investment=5000.0
        ),
        Product(
            ticker="ARKW",
            name="Next Generation Internet ETF",
            risk_score=RiskLevel.AGGRESSIVE,
            asset_class="equity",
            min_investment=3000.0
        ),
        Product(
            ticker="SPXL",
            name="3x Leveraged S&P 500 ETF",
            risk_score=RiskLevel.AGGRESSIVE,
            asset_class="alternative",
            min_investment=10000.0
        ),
    ]
    
    return products


def save_to_json():
    """Save expanded mock data to JSON files in the data/ directory"""
    os.makedirs('data', exist_ok=True)
    
    profiles = generate_investor_profiles(500)
    products = generate_product_master()
    
    # Save profiles
    with open('data/investor_profiles.json', 'w') as f:
        json.dump([p.dict() for p in profiles], f, indent=2)
    
    # Save products
    with open('data/product_master.json', 'w') as f:
        json.dump([p.dict() for p in products], f, indent=2)
        
    print("=" * 60)
    print("✅ EXPANDED MOCK DATA GENERATED INSIDE data/")
    print("  • data/investor_profiles.json")
    print("  • data/product_master.json")
    print("=" * 60)


if __name__ == "__main__":
    save_to_json()
