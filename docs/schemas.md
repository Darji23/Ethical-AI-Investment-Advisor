# 📋 Data Schemas: People (Investors) & Products (ETFs)

This document provides the formal data dictionary, schema constraints, and MongoDB document specifications for both **People (Investor Profiles)** and **Products (ETFs)** in the Ethical AI Investment Advisor.

---

## 🗄️ MongoDB Collections Overview

| Entity | Collection Name | Primary Key | Description | Record Count |
| :--- | :--- | :--- | :--- | :--- |
| **People** | `investor_profiles` | `investor_id` (Unique Index) | Client financial demographics and risk profiles | 5,500 profiles |
| **Products** | `products` | `ticker` (Unique Index) | Real ETF catalog with volatility-based risk metrics (`catalog_v2`) | 197 active ETFs |

---

## 👤 1. People Schema (`InvestorProfile`)

Represents an individual investor seeking automated, personalized investment advice. Used by the RAG context injector to calibrate portfolio suitability.

### Field Definitions

| Field Name | Type | Constraints | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `investor_id` | `string` | Unique, format: `INV[0-9]{3,5}` | Unique investor identifier | `"INV00042"` |
| `name` | `string` | Non-empty | Full name of the client | `"Eleanor Vance"` |
| `age` | `integer` | $18 \le \text{age} \le 100$ | Client age in years | `67` |
| `risk_tolerance` | `integer` (Enum) | $1 \le \text{risk} \le 5$ | Risk tolerance level (1=Conservative to 5=Aggressive) | `2` |
| `annual_income` | `number` (float) | $\ge 0.0$ | Annual earned income in USD | `95000.0` |
| `net_worth` | `number` (float) | $\ge 0.0$ | Total investable net worth in USD | `450000.0` |
| `investment_experience` | `string` (Enum) | `"beginner"`, `"intermediate"`, `"advanced"` | Financial markets sophistication | `"intermediate"` |
| `time_horizon_years` | `integer` | $1 \le \text{horizon} \le 50$ | Intended investment timeframe in years | `8` |

### Risk Tolerance Enum Mapping

| Level | Identifier | Label | Demographic Profile |
| :---: | :--- | :--- | :--- |
| **1** | `CONSERVATIVE` | Conservative | Capital preservation, low volatility, short duration |
| **2** | `MODERATE_CONSERVATIVE` | Moderate Conservative | Income generation with minimal capital drawdown |
| **3** | `MODERATE` | Moderate | Balanced equity and bond exposure |
| **4** | `MODERATE_AGGRESSIVE` | Moderate Aggressive | Long-term growth, high equity allocation |
| **5** | `AGGRESSIVE` | Aggressive | Maximum capital appreciation, high volatility tolerance |

### Validation Rules (Schema Guardrails)
* **Age-Risk Guard:** If `age >= 65`, `risk_tolerance` **cannot** be `5` (`AGGRESSIVE`). It is clamped or rejected by regulatory suitability.
* **Positive Financials:** `annual_income` and `net_worth` cannot be negative.

### Python / Pydantic Model (`app/schemas/schemas.py`)

```python
class InvestorProfile(BaseModel):
    investor_id: str = Field(..., description="Unique identifier")
    name: str = Field(default="", description="Client full name")
    age: int = Field(..., ge=18, le=100, description="Age must be 18-100")
    risk_tolerance: RiskLevel = Field(..., description="Risk tolerance 1-5")
    annual_income: float = Field(..., ge=0, description="Annual income in USD")
    net_worth: float = Field(..., ge=0, description="Net worth in USD")
    investment_experience: Literal["beginner", "intermediate", "advanced"] = Field(...)
    time_horizon_years: int = Field(..., ge=1, le=50, description="Investment horizon")

    @validator('risk_tolerance')
    def validate_risk_for_age(cls, v, values):
        if 'age' in values and values['age'] >= 65 and v >= RiskLevel.AGGRESSIVE:
            raise ValueError(f"Risk level {v} may be unsuitable for age {values['age']} (65+)")
        return v

    class Config:
        use_enum_values = True
        extra = "ignore"
```

### Sample Document in MongoDB (`ethical_advisor.investor_profiles`)

```json
{
  "_id": { "$oid": "664fa7b89c31e210a412c01a" },
  "investor_id": "INV00042",
  "name": "Sarah Jenkins",
  "age": 42,
  "risk_tolerance": 3,
  "annual_income": 115000.0,
  "net_worth": 380000.0,
  "investment_experience": "intermediate",
  "time_horizon_years": 20
}
```

---

## 📈 2. Products Schema (`Product`)

Represents a real Exchange-Traded Fund (ETF) cataloged from live market data via Yahoo Finance (`catalog_v2`). Includes quantitative volatility, maximum drawdown, leverage parameters, and official EU SRRI metrics derived from 5 years of daily trading history.

### Field Definitions

| Field Name | Type | Range / Constraints | Description | Real Example (`SPY`) | Real Example (`TQQQ`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ticker` | `string` | Unique, 1–6 uppercase chars | Stock exchange ticker symbol | `"SPY"` | `"TQQQ"` |
| `name` | `string` | Non-empty | Official legal fund name | `"State Street SPDR S&P 500 ETF Trust"` | `"ProShares UltraPro QQQ"` |
| `risk_score` | `integer` | $1 \le \text{risk} \le 5$ | Regulatory risk rating (VOL-5band-v1; forced 5 for lev/inv) | `3` | `5` |
| `asset_class` | `string` | Holdings taxonomy | Holdings-based asset class | `"equity"` | `"equity"` |
| `share_price` | `number` | $> 0.0$ USD | Latest closing market price per share | `716.70` | `45.28` |
| `price_as_of` | `string` | `YYYY-MM-DD` | Date of the current share price quotation | `"2026-11-26"` | `"2026-11-26"` |
| `description` | `string` | Non-empty (up to 2,000 chars) | Full fund investment mandate & objective | `"The trust seeks to achieve..."` | `"The fund invests in financial..."` |
| `expense_ratio_pct`| `number` | $0.0\% \le \text{fee} \le 3.0\%$ | Net annual management expense ratio in percent | `0.0945` (0.0945%) | `0.78` (0.78%) |
| `volatility_pct` | `number` | $0.0\% < \sigma < 200.0\%$ | Annualized historical daily volatility in percent | `17.56` (17.56%) | `70.43` (70.43%) |
| `max_drawdown_pct`| `number`| $0.0\% \le \text{dd} \le 100.0\%$ | 5-year maximum peak-to-trough decline in percent | `24.50` (24.50%) | `81.26` (81.26%) |
| `fund_family` | `string` | Optional | Fund asset management firm / issuer | `"State Street Investment Management"` | `"ProShares"` |
| `is_leveraged` | `boolean`| `true` or `false` | Indicates whether the fund uses financial leverage | `false` | `true` |
| `is_inverse` | `boolean`| `true` or `false` | Indicates whether the fund short/inverses its index | `false` | `false` |
| `leverage_factor`| `number` | e.g. $-3, -2, -1, 1, 2, 3$ | Leverage multiplier factor | `1.0` | `3.0` |
| `history_years` | `number` | $\ge 3.0$ years | Length of daily adjusted price history analyzed | `4.98` | `4.98` |
| `srri` | `integer` | $1 \le \text{srri} \le 7$ | Official European UCITS SRRI from annualised weekly returns | `6` | `7` |
| `catalog_version` | `string`| e.g. `"catalog_v2"` | Catalog version ID | `"catalog_v2"` | `"catalog_v2"` |
| `active` | `boolean`| `true` or `false` | Flag indicating if fund is actively listed | `true` | `true` |

*(Note: For backward compatibility, `min_investment`, `expense_ratio`, `volatility`, and `max_drawdown` are automatically aliased to the corresponding new fields).*

### Internal Audit Metadata Fields (MongoDB Only)
These fields are saved during ingestion for provenance and compliance audits:
* `_run_id`: Unique ingestion execution ID (e.g. `"run_20261126_054336"`).
* `_fetched_at`: UTC timestamp string when data was downloaded from Yahoo Finance.
* `_data_source`: Data provider identifier (e.g. `"yfinance"`).
* `_risk_method`: Algorithm version used for risk score calculation (`"VOL-5band-v1 (SRRI-inspired)"`).
* `_category`: Yahoo's native fund category (e.g. `"Large Blend"`, `"Ultrashort Bond"`, `"Trading--Leveraged Equity"`).

---

## 🏛️ 3. Asset Class Classification Rules (Holdings-Based)

Asset class is determined strictly by the underlying securities held by the fund, not structural fund mechanics:

| Asset Class | Classification Rule | Benchmark Examples |
| :--- | :--- | :--- |
| **`cash`** | Money market funds and US Treasury funds with **less than 1 year to maturity** | `SHV` (0–1 Year Treasury), `BIL` (1–3 Month T-Bill), `SGOV` |
| **`fixed_income`** | All government, aggregate, corporate, municipal, and inflation-protected bonds with **greater than 1 year maturity** | `BND`, `AGG`, `GOVT`, `LQD`, `HYG`, `TIP` |
| **`equity`** | Broad market, sector, thematic, dividend, international, and **leveraged/inverse equity** (e.g. 3x Nasdaq is `equity` with `leverage_factor: 3.0`, not alternative) | `SPY`, `VOO`, `QQQ`, `IWM`, `EFA`, `TQQQ`, `SQQQ` |
| **`real_estate`** | Real Estate Investment Trusts (REITs) and physical real estate securities | `VNQ`, `IYR`, `XLRE`, `SCHH`, `REET` |
| **`commodity`** | Physical commodities, precious metals, and commodity futures | `GLD` (Gold), `SLV` (Silver), `USO` (Crude Oil), `DBC` |
| **`alternative`** | Crypto & digital assets: pure crypto asset funds (`BITO` active; `IBIT` and `FBTC` rejected by 3-yr rule). Note: thematic funds holding tech equities (`ARKF`) belong in `equity`. | `BITO` (Bitcoin Strategy ETF) |

---

## ⚡ 4. Leverage Detection (Fixes "Ultra" Bug)

Single-word substring matching (like `"Ultra"`, `"Bull"`, `"Bear"`) was removed to avoid misclassifying non-leveraged funds such as ProShares Ultra-Short Bond ETFs.

1. **Category-First Authority:** Leveraged and inverse funds are classified primarily by Yahoo's fund category (`"Trading--Leveraged Equity"`, `"Trading--Inverse Equity"`, `"Trading--Leveraged Debt"`, `"Trading--Inverse Debt"`).
2. **Name Pattern Backup:** Exact regex multiplier tokens (`2X`, `3X`, `-1X`, `-2X`, `-3X`, `"UltraPro"`, `"UltraPro Short"`) are checked.
3. **Disambiguation Rule:** ProShares `"Ultra"` and `"UltraPro"` equity funds are leveraged ($2\times$ and $3\times$), whereas `"Ultra-Short Bond"` or `"Ultrashort Bond"` funds are short-duration fixed income/cash ($1\times$ leverage).
4. **Safety Override:** Any leveraged or inverse product (`is_leveraged == true` or `is_inverse == true`) is **strictly locked to Risk Score 5** regardless of historical volatility.

---

## 📐 5. Volatility-to-Risk Score Calculation (`VOL-5band-v1`)

The risk engine computes annualized historical volatility $\sigma_{\text{ann}}$ from 5-year daily adjusted close prices (excluding splits/dividend jumps):

$$\sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252} \times 100\%$$

| Annualized Volatility ($\sigma_{\text{ann}}$) | Risk Level | Label | Benchmark Funds |
| :---: | :---: | :--- | :--- |
| $\sigma < 5.0\%$ | **1** | Conservative | `SHV` (0.29%), `BIL` (0.25%) |
| $5.0\% \le \sigma < 10.0\%$ | **2** | Moderate Conservative | `BND` (6.06%), `AGG` (6.15%), `GOVT` (5.99%) |
| $10.0\% \le \sigma < 20.0\%$ | **3** | Moderate | `SPY` (17.56%), `VOO` (17.58%), `VTI` (18.15%) |
| $20.0\% \le \sigma < 40.0\%$ | **4** | Moderate Aggressive | `QQQ` (23.76%), `SOXX` (34.92%), `ARKK` (38.87%) |
| $\sigma \ge 40.0\%$ **or** Leveraged/Inverse | **5** | Aggressive | `TQQQ` (70.43%, +3x), `SQQQ` (70.66%, -3x) |

* **Minimum History Guardrail:** Any fund with less than 3.0 years of daily trading history is rejected from ingestion (`FNGU` rejected with 1.77 years).
* **Reference European SRRI (1–7):** Calculated in parallel using weekly Friday closing prices per UCITS guidelines for cross-border regulatory comparison.

---

## 🐍 6. Python / Pydantic Model (`app/schemas/schemas.py`)

```python
class Product(BaseModel):
    ticker: str = Field(..., description="Product ticker symbol")
    name: str = Field(..., description="Product name")
    risk_score: RiskLevel = Field(..., description="Product risk 1-5")
    asset_class: Literal["equity", "fixed_income", "alternative", "cash", "real_estate", "commodity"] = Field(
        ..., description="Holdings-based asset class category"
    )
    share_price: float = Field(..., ge=0, description="Latest share price in USD")
    price_as_of: str = Field(default="", description="Date of price quote (YYYY-MM-DD)")
    description: str = Field(default="", description="Fund objective and mandate description")
    expense_ratio_pct: float = Field(default=0.0, ge=0.0, le=3.0, description="Net expense ratio in %")
    volatility_pct: float = Field(default=0.0, ge=0.0, le=200.0, description="Annualized historical volatility in %")
    max_drawdown_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="5-year maximum drawdown in %")
    fund_family: str = Field(default="", description="Fund issuer/family")
    is_leveraged: bool = Field(default=False, description="Whether fund employs financial leverage")
    is_inverse: bool = Field(default=False, description="Whether fund is an inverse fund")
    leverage_factor: float = Field(default=1.0, description="Leverage multiplier (-3.0 to 3.0)")
    history_years: float = Field(default=0.0, description="Years of historical price data available")
    srri: int = Field(default=0, ge=0, le=7, description="Official EU SRRI (1-7)")
    catalog_version: str = Field(default="catalog_v2", description="Product catalog version")
    active: bool = Field(default=True, description="Active status; false if delisted")

    # Backward-compatible property aliases
    min_investment: Optional[float] = Field(default=None)
    expense_ratio: Optional[float] = Field(default=None)
    volatility: Optional[float] = Field(default=None)
    max_drawdown: Optional[float] = Field(default=None)
```

---

## 🔍 7. Real Sample MongoDB Documents

### Benchmark Equity: `SPY`
```json
{
  "_id": { "$oid": "6b076d8b1f994995f61c2b53" },
  "ticker": "SPY",
  "name": "State Street SPDR S&P 500 ETF Trust",
  "risk_score": 3,
  "asset_class": "equity",
  "share_price": 716.7,
  "price_as_of": "2026-11-26",
  "description": "The trust seeks to achieve its investment objective by holding a portfolio of the common stocks that are included in the index (the “Portfolio”), with the weight of each stock in the Portfolio substantially corresponding to the weight of such stock in the index.",
  "expense_ratio_pct": 0.0945,
  "volatility_pct": 17.56,
  "max_drawdown_pct": 24.5,
  "fund_family": "State Street Investment Management",
  "is_leveraged": false,
  "is_inverse": false,
  "leverage_factor": 1.0,
  "history_years": 4.98,
  "srri": 6,
  "catalog_version": "catalog_v2",
  "active": true,
  "min_investment": 716.7,
  "_run_id": "run_20261126_054336",
  "_category": "Large Blend",
  "_fetched_at": "2026-11-26T05:43:42.664031+00:00",
  "_data_source": "yfinance",
  "_risk_method": "VOL-5band-v1 (SRRI-inspired)"
}
```

### Leveraged Equity: `TQQQ`
```json
{
  "_id": { "$oid": "6b076d911f994995f61c2c03" },
  "ticker": "TQQQ",
  "name": "ProShares UltraPro QQQ",
  "risk_score": 5,
  "asset_class": "equity",
  "share_price": 45.28,
  "price_as_of": "2026-11-26",
  "description": "The fund invests in financial instruments that ProShare Advisors believes, in combination, should produce daily returns consistent with the Daily Target. The index is designed to measure the performance of 100 of the largest Nasdaq-listed non-financial companies...",
  "expense_ratio_pct": 0.78,
  "volatility_pct": 70.43,
  "max_drawdown_pct": 81.26,
  "fund_family": "ProShares",
  "is_leveraged": true,
  "is_inverse": false,
  "leverage_factor": 3.0,
  "history_years": 4.98,
  "srri": 7,
  "catalog_version": "catalog_v2",
  "active": true,
  "min_investment": 45.28,
  "_run_id": "run_20261126_054336",
  "_category": "Trading--Leveraged Equity",
  "_fetched_at": "2026-11-26T05:47:58.743634+00:00",
  "_data_source": "yfinance",
  "_risk_method": "VOL-5band-v1 (SRRI-inspired)"
}
```

---

## 🔗 8. Unified Regulatory & Firm Policy Suitability Rule Table (7 Canonical Rules)

All suitability checks are defined in a single declarative rule table in [app/services/suitability_validator.py](../app/services/suitability_validator.py). Both `get_suitable_products(profile)` and `validate_recommendation(profile, recommendation)` evaluate this exact shared table.

### Regulatory Standards & The Safe Pattern
* **Governing Standard for Retail Recommendations:** **SEC Regulation Best Interest (SEA Rule 15l-1, "Reg BI")**. FINRA Rule 2111 does not apply to recommendations subject to Reg BI; for retail clients, Reg BI's Care Obligation governs, while FINRA 2111's list of customer profile factors remains the industry benchmark for what customer information must be considered.
* **The Safe Regulatory Pattern:** Give the authoritative regulatory source for the *factor*, and identify specific numeric thresholds as *Firm Policy*.

| Rule ID | Rule Name | Regulatory Factor Source | Firm Policy Implementation & Threshold |
| :--- | :--- | :--- | :--- |
| **`RULE_1_LEVERAGE`** | **Non-Traditional (Leveraged & Inverse) ETF Exclusion** | **FINRA Regulatory Notice 09-31** & **FINRA Regulatory Notice 12-03** (Complex Products) | **Firm Policy: 0% Allocation.** System provides buy-and-hold advice ($H \ge 1$ yr). Leveraged ($>1\times$) and inverse ($<0\times$) ETFs reset daily and suffer compounding volatility decay; they are **never recommended** in client portfolios. *(Note: While FINRA does not prohibit them for every customer in day trading, blocking them for long-term advice is our firm's conservative choice).* |
| **`RULE_2_TIME_HORIZON`** | **Customer Time Horizon & Capital Preservation** | **Reg BI Care Obligation (SEA Rule 15l-1)**; Customer Time Horizon factor under **FINRA Rule 2111** | **Firm Policy:** For short horizons ($< 3$ years): products with `risk_score >= 4` or `asset_class == 'alternative'` are prohibited; portfolio must hold $\ge 50\%$ in `fixed_income` or `cash`. |
| **`RULE_3_RISK_ALIGNMENT`** | **Customer Risk Tolerance Alignment** | **Reg BI Care Obligation (SEA Rule 15l-1)**; Customer Risk Tolerance factor under **FINRA Rule 2111** | **Firm Policy:** Individual products cannot exceed customer tolerance by more than 1 level ($R_p \le R_i + 1$; soft disclosure warning if $R_p = R_i + 1$). Weighted average portfolio risk must not exceed $R_i + 0.50$. |
| **`RULE_4_SENIOR_PROTECTION`** | **Senior Investor Risk Capping** | Customer Age factor under **FINRA Rule 2111** & **Reg BI Care Obligation** | **Firm Policy:** Age $70+$ cannot hold products with `risk_score >= 4` (capped at Risk 3). Age $65-69$ cannot hold aggressive `risk_score == 5` products (capped at Risk 4). Age $60-64$ receives risk disclosure warnings on Risk 5. *(Note: Not FINRA 2165, which addresses exploitation/temporary disbursement holds).* |
| **`RULE_5_EXPERIENCE_COMPLEXITY`** | **Investor Experience & Complex Products** | **FINRA Regulatory Notice 12-03** (Heightened Supervision of Complex Products) & **Reg BI Care Obligation** | **Firm Policy:** Beginners cannot access aggressive Risk 5 products, alternative/crypto assets, or futures-based derivative commodities (`USO`, `DBC`, etc.) that suffer contango decay. Plain physical precious metals (`GLD`, `IAU`) are permitted. Intermediates require risk disclosure on Risk 5. |
| **`RULE_6_CONCENTRATION_LIMITS`** | **Portfolio Diversification & Single-Product Caps** | Prudent portfolio diversification standards under the **Reg BI Care Obligation** | **Firm Policy:** Single product allocation cannot exceed $60\%$. Single asset class allocation $> 80\%$ triggers a mandatory concentration warning. *(Note: No FINRA notice sets specific numeric percentages).* |
| **`RULE_7_FINANCIAL_CAPACITY`** | **Share Price Affordability & Minimum Investment** | Customer Financial Situation and Needs factor under **FINRA Rule 2111** & **Reg BI Care Obligation** | **Firm Policy:** Allocated dollar amount (assumed from $10\%$ of client net worth) must meet or exceed single-share market purchase price (`share_price`). |
