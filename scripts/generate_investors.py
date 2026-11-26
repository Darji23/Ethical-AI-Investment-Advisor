"""
Investor Profile Generator — synthetic-v2

Generates 5,500 realistic investor profiles using believable correlated
distributions, plus a mandatory set of labelled edge-case profiles.

Distributional assumptions (not from real data unless stated):
- Income: lognormal, skewed toward middle incomes with a long right tail.
  Loosely calibrated to U.S. household income distributions
  (Federal Reserve Survey of Consumer Finances, 2022).
- Investable assets: grow with age and income, noisy.
- Time horizon: shrinks with age, with noise.
- Risk tolerance: tends to fall with age, rise with experience; noise ensures
  unusual combinations still occur.
- Experience: tends to grow with age, with noise.
All parameters are assumptions; they are NOT taken directly from SCF data.

Edge cases are hard-coded and tagged with segment="edge_case".
Contradictory profiles receive a profile_warnings list.
"""

import random
import math
from datetime import date
from typing import List
from pymongo import MongoClient, ReplaceOne

GENERATOR_TAG = "synthetic-v2"
SEED = 42
TOTAL_INVESTORS = 5500  # INV00001 – INV05500

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lognormal_income(rng: random.Random, age: int, experience: str) -> float:
    """
    Annual gross income drawn from a lognormal distribution.
    mu / sigma loosely match U.S. household income shape (SCF 2022).
    """
    base_mu = 10.8   # exp(10.8) ≈ $49k median
    if experience == "advanced":
        base_mu += 0.6
    elif experience == "intermediate":
        base_mu += 0.3
    # Peak earnings mid-career
    age_bonus = max(0.0, 0.015 * (min(age, 55) - 25))
    mu = base_mu + age_bonus
    sigma = 0.70
    raw = math.exp(rng.gauss(mu, sigma))
    return max(18_000, min(raw, 800_000))


def _investable_assets(rng: random.Random, income: float, age: int) -> float:
    """
    Investable assets grow with age and savings rate; lots of noise.
    Rule of thumb: by 40 people have ~1–3× income saved (varies hugely).
    """
    years_saving = max(0, age - 22)
    savings_rate = rng.uniform(0.03, 0.20)   # 3–20% of income per year
    base = income * savings_rate * years_saving
    noise = rng.uniform(0.4, 1.8)
    raw = base * noise
    return max(0, min(raw, 10_000_000))


def _net_worth(rng: random.Random, investable: float, income: float, age: int) -> float:
    """Net worth = investable + illiquid (home equity, etc)."""
    illiquid_multiple = rng.uniform(0, 2.5) if age > 30 else rng.uniform(0, 0.5)
    illiquid = income * illiquid_multiple
    return investable + illiquid


def _time_horizon(rng: random.Random, age: int) -> int:
    """Horizon shrinks with age; retirement target ~65."""
    base = max(1, 65 - age)
    jitter = rng.randint(-5, 10)
    return max(1, min(50, base + jitter))


def _risk_tolerance(rng: random.Random, age: int, experience: str) -> int:
    """
    1–5 risk score. Tends to fall with age, rise with experience.
    Noise means extremes still occur at any age.
    """
    if experience == "advanced":
        center = 3.8
    elif experience == "intermediate":
        center = 3.2
    else:
        center = 2.5
    # Age adjustment: each decade over 30 nudges toward conservative
    age_adj = -0.04 * max(0, age - 30)
    mu = center + age_adj
    raw = rng.gauss(mu, 1.0)
    return max(1, min(5, round(raw)))


def _experience(rng: random.Random, age: int) -> str:
    """Experience tends to grow with age, but young experts and old beginners exist."""
    if age < 28:
        weights = [0.65, 0.30, 0.05]
    elif age < 45:
        weights = [0.30, 0.45, 0.25]
    else:
        weights = [0.15, 0.40, 0.45]
    return rng.choices(["beginner", "intermediate", "advanced"], weights=weights)[0]


def _objective(rng: random.Random, risk: int, age: int) -> str:
    """Investment objective correlated with risk tolerance and age."""
    if risk == 1:
        return rng.choices(["preservation", "income"], weights=[0.7, 0.3])[0]
    elif risk == 2:
        return rng.choices(["preservation", "income", "balanced"], weights=[0.3, 0.4, 0.3])[0]
    elif risk == 3:
        return rng.choices(["income", "balanced", "growth"], weights=[0.2, 0.5, 0.3])[0]
    elif risk == 4:
        return rng.choices(["balanced", "growth", "speculation"], weights=[0.2, 0.6, 0.2])[0]
    else:
        return rng.choices(["growth", "speculation"], weights=[0.4, 0.6])[0]


def _liquidity(rng: random.Random, horizon: int, risk: int) -> str:
    if horizon <= 2 or risk <= 2:
        return rng.choices(["low", "medium", "high"], weights=[0.2, 0.4, 0.4])[0]
    elif horizon <= 5:
        return rng.choices(["low", "medium", "high"], weights=[0.3, 0.5, 0.2])[0]
    else:
        return rng.choices(["low", "medium", "high"], weights=[0.5, 0.4, 0.1])[0]


def _detect_warnings(age: int, risk: int, objective: str, liquidity: str, horizon: int) -> List[str]:
    """Flag contradictory profile combinations (not errors — just noteworthy)."""
    warnings = []
    if risk == 1 and objective == "speculation":
        warnings.append("Contradiction: risk_tolerance=1 (conservative) conflicts with objective=speculation")
    if risk == 5 and horizon <= 2:
        warnings.append("Contradiction: risk_tolerance=5 (aggressive) with time_horizon≤2 years")
    if risk == 5 and objective == "preservation":
        warnings.append("Contradiction: risk_tolerance=5 conflicts with objective=preservation")
    if liquidity == "high" and risk >= 4:
        warnings.append("Note: high liquidity_needs with aggressive risk tolerance — validator will apply extra constraints")
    if age >= 70 and risk == 5:
        warnings.append("Note: age≥70 with aggressive risk tolerance — suitability rules apply extra scrutiny")
    return warnings


# ---------------------------------------------------------------------------
# edge cases (hard-coded, always included)
# ---------------------------------------------------------------------------

EDGE_CASES = [
    # 1. 75-year-old aggressive investor
    dict(investor_id="INV00001", age=75, risk_tolerance=5,
         annual_income=95_000, net_worth=1_200_000, investable_assets=800_000,
         investment_experience="advanced", time_horizon_years=10,
         investment_objective="growth", liquidity_needs="low",
         segment="edge_case"),

    # 2. 22-year-old beginner with $2,000 to invest
    dict(investor_id="INV00002", age=22, risk_tolerance=3,
         annual_income=28_000, net_worth=3_500, investable_assets=2_000,
         investment_experience="beginner", time_horizon_years=10,
         investment_objective="growth", liquidity_needs="medium",
         segment="edge_case"),

    # 3. 1-year horizon wanting aggressive growth (contradictory)
    dict(investor_id="INV00003", age=35, risk_tolerance=5,
         annual_income=120_000, net_worth=250_000, investable_assets=80_000,
         investment_experience="intermediate", time_horizon_years=1,
         investment_objective="speculation", liquidity_needs="high",
         segment="edge_case"),

    # 4. High income, almost no investable assets
    dict(investor_id="INV00004", age=42, risk_tolerance=3,
         annual_income=320_000, net_worth=180_000, investable_assets=5_000,
         investment_experience="intermediate", time_horizon_years=15,
         investment_objective="balanced", liquidity_needs="medium",
         segment="edge_case"),

    # 5. Age exactly 64 (just below senior threshold)
    dict(investor_id="INV00005", age=64, risk_tolerance=5,
         annual_income=85_000, net_worth=600_000, investable_assets=350_000,
         investment_experience="advanced", time_horizon_years=5,
         investment_objective="growth", liquidity_needs="low",
         segment="edge_case"),

    # 6. Age exactly 65 (senior threshold)
    dict(investor_id="INV00006", age=65, risk_tolerance=5,
         annual_income=70_000, net_worth=550_000, investable_assets=300_000,
         investment_experience="advanced", time_horizon_years=5,
         investment_objective="growth", liquidity_needs="low",
         segment="edge_case"),

    # 7. Age exactly 69 (just below stricter senior cap)
    dict(investor_id="INV00007", age=69, risk_tolerance=4,
         annual_income=65_000, net_worth=480_000, investable_assets=260_000,
         investment_experience="intermediate", time_horizon_years=8,
         investment_objective="balanced", liquidity_needs="medium",
         segment="edge_case"),

    # 8. Age exactly 70 (stricter senior cap kicks in)
    dict(investor_id="INV00008", age=70, risk_tolerance=4,
         annual_income=60_000, net_worth=420_000, investable_assets=220_000,
         investment_experience="intermediate", time_horizon_years=8,
         investment_objective="income", liquidity_needs="medium",
         segment="edge_case"),

    # 9. Conservative risk + speculation objective (contradictory)
    dict(investor_id="INV00009", age=50, risk_tolerance=1,
         annual_income=100_000, net_worth=400_000, investable_assets=150_000,
         investment_experience="beginner", time_horizon_years=12,
         investment_objective="speculation", liquidity_needs="low",
         segment="edge_case"),

    # 10. Aggressive risk + 1-year horizon (contradictory)
    dict(investor_id="INV00010", age=28, risk_tolerance=5,
         annual_income=55_000, net_worth=30_000, investable_assets=15_000,
         investment_experience="beginner", time_horizon_years=1,
         investment_objective="speculation", liquidity_needs="high",
         segment="edge_case"),
]

NUM_EDGE_CASES = len(EDGE_CASES)


# ---------------------------------------------------------------------------
# main generator
# ---------------------------------------------------------------------------

def generate_investors(total: int = TOTAL_INVESTORS) -> List[dict]:
    rng = random.Random(SEED)
    profiles = []

    # -- Edge cases first (IDs INV00001 – INV00010) -------------------------
    for ec in EDGE_CASES:
        warnings = _detect_warnings(
            ec["age"], ec["risk_tolerance"],
            ec["investment_objective"], ec["liquidity_needs"],
            ec["time_horizon_years"]
        )
        doc = {**ec,
               "profile_warnings": warnings,
               "_generator": GENERATOR_TAG,
               "_seed": SEED,
               "_generated_on": str(date.today())}
        profiles.append(doc)

    # -- Synthetic profiles (IDs INV00011 onwards) --------------------------
    remaining = total - NUM_EDGE_CASES
    for i in range(remaining):
        idx = NUM_EDGE_CASES + i + 1
        investor_id = f"INV{idx:05d}"

        age = int(rng.triangular(18, 80, 42))  # skewed toward mid-career
        experience = _experience(rng, age)
        risk = _risk_tolerance(rng, age, experience)
        income = _lognormal_income(rng, age, experience)
        investable = _investable_assets(rng, income, age)
        net_worth = _net_worth(rng, investable, income, age)
        horizon = _time_horizon(rng, age)
        objective = _objective(rng, risk, age)
        liquidity = _liquidity(rng, horizon, risk)
        warnings = _detect_warnings(age, risk, objective, liquidity, horizon)

        doc = {
            "investor_id": investor_id,
            "age": age,
            "risk_tolerance": risk,
            "annual_income": round(income, 2),
            "net_worth": round(net_worth, 2),
            "investable_assets": round(investable, 2),
            "investment_experience": experience,
            "time_horizon_years": horizon,
            "investment_objective": objective,
            "liquidity_needs": liquidity,
            "segment": None,
            "profile_warnings": warnings,
            "_generator": GENERATOR_TAG,
            "_seed": SEED,
            "_generated_on": str(date.today()),
        }
        profiles.append(doc)

    return profiles


# ---------------------------------------------------------------------------
# summary printer
# ---------------------------------------------------------------------------

def print_summary(profiles: List[dict]):
    n = len(profiles)
    print(f"\n{'='*60}")
    print(f"  INVESTOR GENERATOR SUMMARY  ({GENERATOR_TAG}, seed={SEED})")
    print(f"{'='*60}")
    print(f"  Total profiles:      {n}")
    print(f"  Edge cases:          {sum(1 for p in profiles if p.get('segment') == 'edge_case')}")
    print(f"  With warnings:       {sum(1 for p in profiles if p.get('profile_warnings'))}")

    ages = [p['age'] for p in profiles]
    print(f"\n  Age:  min={min(ages)}, max={max(ages)}, "
          f"mean={sum(ages)/n:.1f}")

    for label, field, vals in [
        ("Risk tolerance", "risk_tolerance", range(1,6)),
        ("Experience",     "investment_experience", ["beginner","intermediate","advanced"]),
        ("Objective",      "investment_objective",  ["preservation","income","balanced","growth","speculation"]),
        ("Liquidity",      "liquidity_needs",       ["low","medium","high"]),
    ]:
        print(f"\n  {label}:")
        for v in vals:
            c = sum(1 for p in profiles if p[field] == v)
            bar = "█" * int(c * 40 / n)
            print(f"    {str(v):15s}: {c:5d} ({c/n*100:4.1f}%) {bar}")

    horizons = [p['time_horizon_years'] for p in profiles]
    print(f"\n  Time horizon: min={min(horizons)}, max={max(horizons)}, "
          f"mean={sum(horizons)/n:.1f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# seed MongoDB
# ---------------------------------------------------------------------------

def seed_mongodb(profiles: List[dict]):
    from app.core.config import MONGODB_URL, MONGODB_DB_NAME
    client = MongoClient(MONGODB_URL)
    db = client[MONGODB_DB_NAME]
    col = db["investor_profiles"]

    # Ensure unique index on investor_id
    col.create_index("investor_id", unique=True)

    # Remove stale pre-v2 records (e.g. INV001-INV500 without _generator="synthetic-v2")
    deleted = col.delete_many({"_generator": {"$ne": GENERATOR_TAG}})
    if deleted.deleted_count:
        print(f"🧹 Removed {deleted.deleted_count} stale pre-{GENERATOR_TAG} investor records")

    ops = [
        ReplaceOne({"investor_id": p["investor_id"]}, p, upsert=True)
        for p in profiles
    ]
    result = col.bulk_write(ops, ordered=False)
    client.close()
    print(f"✅ MongoDB: upserted {result.upserted_count + result.modified_count} "
          f"investor profiles into '{MONGODB_DB_NAME}.investor_profiles'")


if __name__ == "__main__":
    profiles = generate_investors(TOTAL_INVESTORS)
    print_summary(profiles)
    seed_mongodb(profiles)
