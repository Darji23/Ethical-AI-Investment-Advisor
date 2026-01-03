"""
Generate expanded mock datasets for testing the suitability engine
100 Investor Profiles + 25 Investment Products
"""
import json
import random
from typing import List
from schemas import InvestorProfile, Product, RiskLevel

# Set seed for reproducibility
random.seed(42)

def generate_investor_profiles(count: int = 100) -> List[InvestorProfile]:
    """
    Create 100 diverse investor profiles across different demographics
    
    Distribution:
    - Age: Uniform across 22-75 years
    - Risk Tolerance: Varied based on age and experience
    - Income: $35k - $500k
    - Net Worth: $10k - $5M
    - Experience: Beginner (30%), Intermediate (50%), Advanced (20%)
    - Time Horizon: 5-40 years
    """
    profiles = []
    
    # Experience level distribution
    experience_levels = (
        ["beginner"] * 30 + 
        ["intermediate"] * 50 + 
        ["advanced"] * 20
    )
    random.shuffle(experience_levels)
    
    for i in range(count):
        investor_id = f"INV{i+1:03d}"  # INV001 to INV100
        
        # Age distribution (22-75)
        age = random.randint(22, 75)
        
        # Risk tolerance based on age (younger = more aggressive)
        if age < 30:
            risk_weights = [0.05, 0.15, 0.30, 0.35, 0.15]  # Favor aggressive
        elif age < 45:
            risk_weights = [0.10, 0.20, 0.40, 0.20, 0.10]  # Favor moderate
        elif age < 60:
            risk_weights = [0.20, 0.35, 0.30, 0.10, 0.05]  # Favor conservative
        else:
            risk_weights = [0.40, 0.35, 0.20, 0.05, 0.00]  # Very conservative
        
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
        age_multiplier = (age - 20) / 10  # 0.2 to 5.5
        base_net_worth = annual_income * random.uniform(1, 4) * age_multiplier
        net_worth = max(10000, min(base_net_worth, 5000000))  # $10k to $5M
        
        # Time horizon (inversely correlated with age)
        time_horizon_years = max(5, min(40, 75 - age + random.randint(-5, 10)))
        
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
    
    Distribution:
    - Conservative (Risk 1): 5 products
    - Moderate Conservative (Risk 2): 5 products
    - Moderate (Risk 3): 7 products
    - Moderate Aggressive (Risk 4): 5 products
    - Aggressive (Risk 5): 3 products
    
    Asset Classes: Fixed Income, Equity, Alternatives, Real Estate, Commodities
    """
    products = [
        # ============================================
        # CONSERVATIVE PRODUCTS (Risk Level 1)
        # ============================================
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
        
        # ============================================
        # MODERATE CONSERVATIVE PRODUCTS (Risk Level 2)
        # ============================================
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
        
        # ============================================
        # MODERATE PRODUCTS (Risk Level 3)
        # ============================================
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
        
        # ============================================
        # MODERATE AGGRESSIVE PRODUCTS (Risk Level 4)
        # ============================================
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
        
        # ============================================
        # AGGRESSIVE PRODUCTS (Risk Level 5)
        # ============================================
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
    """Save expanded mock data to JSON files"""
    profiles = generate_investor_profiles(100)
    products = generate_product_master()
    
    # Save profiles
    with open('investor_profiles.json', 'w') as f:
        json.dump([p.dict() for p in profiles], f, indent=2)
    
    # Save products
    with open('product_master.json', 'w') as f:
        json.dump([p.dict() for p in products], f, indent=2)
    
    # Generate statistics
    age_distribution = {
        '20-30': sum(1 for p in profiles if 20 <= p.age < 30),
        '30-40': sum(1 for p in profiles if 30 <= p.age < 40),
        '40-50': sum(1 for p in profiles if 40 <= p.age < 50),
        '50-60': sum(1 for p in profiles if 50 <= p.age < 60),
        '60-70': sum(1 for p in profiles if 60 <= p.age < 70),
        '70+': sum(1 for p in profiles if p.age >= 70),
    }
    
    risk_distribution = {
        'Conservative': sum(1 for p in profiles if p.risk_tolerance == RiskLevel.CONSERVATIVE),
        'Moderate Conservative': sum(1 for p in profiles if p.risk_tolerance == RiskLevel.MODERATE_CONSERVATIVE),
        'Moderate': sum(1 for p in profiles if p.risk_tolerance == RiskLevel.MODERATE),
        'Moderate Aggressive': sum(1 for p in profiles if p.risk_tolerance == RiskLevel.MODERATE_AGGRESSIVE),
        'Aggressive': sum(1 for p in profiles if p.risk_tolerance == RiskLevel.AGGRESSIVE),
    }
    
    experience_distribution = {
        'Beginner': sum(1 for p in profiles if p.investment_experience == "beginner"),
        'Intermediate': sum(1 for p in profiles if p.investment_experience == "intermediate"),
        'Advanced': sum(1 for p in profiles if p.investment_experience == "advanced"),
    }
    
    product_distribution = {
        'Conservative': sum(1 for p in products if p.risk_score == RiskLevel.CONSERVATIVE),
        'Moderate Conservative': sum(1 for p in products if p.risk_score == RiskLevel.MODERATE_CONSERVATIVE),
        'Moderate': sum(1 for p in products if p.risk_score == RiskLevel.MODERATE),
        'Moderate Aggressive': sum(1 for p in products if p.risk_score == RiskLevel.MODERATE_AGGRESSIVE),
        'Aggressive': sum(1 for p in products if p.risk_score == RiskLevel.AGGRESSIVE),
    }
    
    print("=" * 60)
    print("✅ EXPANDED MOCK DATA GENERATED")
    print("=" * 60)
    print(f"\n📊 INVESTOR PROFILES: {len(profiles)}")
    print("\nAge Distribution:")
    for age_range, count in age_distribution.items():
        print(f"  {age_range}: {count} ({count}%)")
    
    print("\nRisk Tolerance Distribution:")
    for risk_level, count in risk_distribution.items():
        print(f"  {risk_level}: {count} ({count}%)")
    
    print("\nExperience Distribution:")
    for exp_level, count in experience_distribution.items():
        print(f"  {exp_level}: {count} ({count}%)")
    
    print(f"\n💼 INVESTMENT PRODUCTS: {len(products)}")
    print("\nProduct Risk Distribution:")
    for risk_level, count in product_distribution.items():
        print(f"  {risk_level}: {count} products")
    
    print("\n📁 Files saved:")
    print("  • investor_profiles.json")
    print("  • product_master.json")
    print("=" * 60)


if __name__ == "__main__":
    save_to_json()