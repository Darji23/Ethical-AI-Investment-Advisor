"""
Real Data Ingestion Script — ETF Products & Synthetic Investors
=============================================================
Fetches real ETF data from Yahoo Finance, detects leverage/inverse flags,
calculates volatility-based risk scores ("VOL-5band-v1 (SRRI-inspired)"),
computes official EU SRRI (1-7), validates ranges, and stores data in MongoDB.

Usage:
    python3 -m scripts.ingest_data              # Full run (products + investors + FAISS)
    python3 -m scripts.ingest_data --products    # Products only
    python3 -m scripts.ingest_data --investors   # Investors only
    python3 -m scripts.ingest_data --rebuild     # Rebuild FAISS index only
    python3 -m scripts.ingest_data --resume      # Resume interrupted product fetch
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Curated ETF ticker universe (~200 tickers across all asset classes)
# ---------------------------------------------------------------------------
ETF_TICKERS: List[str] = [
    # ── Broad US Equity ──────────────────────────────────────────────────
    "SPY", "VOO", "VTI", "IVV", "SPYM", "SCHB", "ITOT", "SPTM",
    "RSP", "SCHX",
    # ── US Mid / Small Cap ───────────────────────────────────────────────
    "IJR", "IJH", "VB", "VXF", "SCHA", "MDYG", "IWM", "IWO",
    "IWN", "SLYG",
    # ── Nasdaq / Growth / Value ──────────────────────────────────────────
    "QQQ", "QQQM", "VUG", "VTV", "VOOG", "VOOV", "IWF", "IWD",
    "MGK", "SCHG",
    # ── Dividend / Income Equity ─────────────────────────────────────────
    "VIG", "VYM", "SCHD", "DVY", "SDY", "DGRO", "HDV", "NOBL",
    "SPYD", "DGRW",
    # ── US Sectors ───────────────────────────────────────────────────────
    "XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLU",
    "XLC", "XLB", "XLRE",
    "VGT", "VFH", "VHT", "VDE", "VCR", "VDC", "VIS", "VPU",
    "VOX",
    # ── Thematic / Sector Niche ──────────────────────────────────────────
    "ARKK", "ARKW", "ARKG", "ARKF",
    "SOXX", "SMH", "HACK", "BOTZ", "ROBO", "SKYY",
    "IBB", "XBI", "GNOM",
    "TAN", "ICLN", "QCLN", "PBW",
    "LIT", "REMX",
    # ── ESG / Sustainable ────────────────────────────────────────────────
    "ESGU", "ESGV", "SUSA", "DSI", "SNPE", "VEGN",
    "NULG", "NULV", "NACP", "EFIV", "SDG",
    # ── International Developed ──────────────────────────────────────────
    "VXUS", "VEA", "IEFA", "EFA", "IXUS", "SCHF",
    "VGK", "EWJ", "EWG", "EWU", "EWQ", "EWC", "EWA",
    # ── Emerging Markets ─────────────────────────────────────────────────
    "VWO", "EEM", "IEMG", "SCHE",
    "FXI", "MCHI", "EWZ", "INDA", "EWT", "EWY",
    "KWEB", "GEM",
    # ── Cash / Money Market Equivalents (<1 Year Maturity) ──────────────
    "SHV", "BIL", "SGOV",
    # ── US Treasuries (>1 Year Maturity) ─────────────────────────────────
    "SHY", "IEI", "IEF", "TLT", "GOVT", "VGSH", "VGIT",
    "VGLT", "SCHO", "SCHR",
    # ── US Aggregate / Investment-Grade Bonds ────────────────────────────
    "BND", "AGG", "SCHZ", "IUSB", "BSV", "BIV", "BLV",
    # ── Corporate Bonds ──────────────────────────────────────────────────
    "LQD", "VCIT", "VCSH", "VCLT", "IGIB",
    # ── High Yield Bonds ─────────────────────────────────────────────────
    "HYG", "JNK", "SHYG", "USHY",
    # ── TIPS / Inflation-Protected ───────────────────────────────────────
    "TIP", "VTIP", "SCHP", "STIP",
    # ── Municipal Bonds ──────────────────────────────────────────────────
    "MUB", "VTEB", "TFI",
    # ── International Bonds ──────────────────────────────────────────────
    "BNDX", "IAGG", "EMB", "VWOB",
    # ── Real Estate ──────────────────────────────────────────────────────
    "VNQ", "VNQI", "SCHH", "IYR", "RWR", "REET",
    # ── Commodities ──────────────────────────────────────────────────────
    "GLD", "IAU", "SLV", "PPLT", "GDX", "GDXJ",
    "USO", "BNO", "DBA", "DBC", "PDBC", "GSG",
    "CPER",
    # ── Multi-Asset / Balanced ───────────────────────────────────────────
    "AOR", "AOM", "AOA", "AOK",
    # ── Crypto / Digital Assets ──────────────────────────────────────────
    "BITO", "IBIT", "FBTC",
    # ── Leveraged (2x, 3x) ───────────────────────────────────────────────
    "TQQQ", "SPXL", "UPRO", "SOXL", "TECL", "FAS", "LABU",
    "TNA", "UDOW", "FNGU",
    # ── Inverse (-1x, -2x, -3x) ──────────────────────────────────────────
    "SQQQ", "SPXS", "SDS", "SH", "PSQ", "DOG", "SOXS",
    "FNGD", "FAZ", "SRTY",
]

# Remove duplicates while preserving order
ETF_TICKERS = list(dict.fromkeys(ETF_TICKERS))

# ---------------------------------------------------------------------------
# 1. LEVERAGE & INVERSE DETECTION (Fixes "Ultra" Bug)
# ---------------------------------------------------------------------------
def detect_leverage(category: str, name: str, ticker: str) -> Tuple[bool, bool, float]:
    """
    Detect whether a fund is leveraged or inverse, and its leverage multiplier.
    
    Rules:
    1. Category is authoritative:
       - 'Trading--Leveraged Equity/Debt/Commodities' -> is_leveraged=True
       - 'Trading--Inverse Equity/Debt/Commodities'   -> is_inverse=True
    2. ProShares 'Ultra' and 'UltraPro' funds are leveraged (2x and 3x).
       'Ultra-Short Bond' funds are NOT leveraged (fixed-income short maturity).
    3. As backup, match precise patterns: 2X, 3X, -1X, -2X, -3X.
    
    Returns:
        (is_leveraged: bool, is_inverse: bool, leverage_factor: float)
    """
    cat = (category or "").lower()
    nm = (name or "").lower()
    tick = (ticker or "").upper()
    
    is_leveraged = False
    is_inverse = False
    leverage_factor = 1.0
    
    # Check if this is an ultra-short bond/debt product (NOT leveraged)
    is_ultrashort_debt = (
        ("ultrashort" in cat or "ultra short" in cat or "ultra-short" in cat)
        and any(w in cat or w in nm for w in ["bond", "debt", "treasury", "government"])
    )
    
    # 1. Category primary check
    if "trading--leveraged" in cat or "leveraged equity" in cat or "leveraged debt" in cat or "leveraged commodit" in cat:
        is_leveraged = True
        if "3x" in nm or "3x" in tick or "ultrapro" in nm or "triple" in nm:
            leverage_factor = 3.0
        else:
            leverage_factor = 2.0
            
    elif "trading--inverse" in cat or "inverse equity" in cat or "inverse debt" in cat or "inverse commodit" in cat:
        is_inverse = True
        if "3x" in nm or "3x" in tick or "ultrapro short" in nm:
            is_leveraged = True
            leverage_factor = -3.0
        elif "2x" in nm or "2x" in tick or ("ultrashort" in nm and not is_ultrashort_debt):
            is_leveraged = True
            leverage_factor = -2.0
        else:
            is_leveraged = False
            leverage_factor = -1.0
            
    # 2. Backup pattern matching in name or ticker (excluding ultra-short debt)
    elif not is_ultrashort_debt:
        # Match explicit 3X / -3X
        if re.search(r'(?:\b|_)3x(?:\b|_)', nm, re.I) or re.search(r'\b3x\b', tick, re.I):
            if any(w in nm for w in ["bear", "short", "inverse", "-3x"]):
                is_inverse = True
                is_leveraged = True
                leverage_factor = -3.0
            else:
                is_leveraged = True
                leverage_factor = 3.0
                
        # Match explicit 2X / -2X
        elif re.search(r'(?:\b|_)2x(?:\b|_)', nm, re.I) or re.search(r'\b2x\b', tick, re.I):
            if any(w in nm for w in ["bear", "short", "inverse", "-2x"]):
                is_inverse = True
                is_leveraged = True
                leverage_factor = -2.0
            else:
                is_leveraged = True
                leverage_factor = 2.0
                
        # Match explicit -1X
        elif re.search(r'(?:\b|_)-1x(?:\b|_)', nm, re.I) or ("proshares short" in nm and "ultra" not in nm):
            is_inverse = True
            leverage_factor = -1.0
            
        # ProShares specific naming rules
        elif "ultrapro" in nm:
            if "short" in nm:
                is_inverse = True
                is_leveraged = True
                leverage_factor = -3.0
            else:
                is_leveraged = True
                leverage_factor = 3.0
        elif "proshares ultra " in nm and "short" not in nm:
            is_leveraged = True
            leverage_factor = 2.0
        elif "proshares ultrashort " in nm or "proshares ultra short " in nm:
            is_inverse = True
            is_leveraged = True
            leverage_factor = -2.0

    return is_leveraged, is_inverse, float(leverage_factor)


# ---------------------------------------------------------------------------
# 2. ASSET CLASS CLASSIFIER (Holdings-Based)
# ---------------------------------------------------------------------------
def category_to_asset_class(category: str, name: str = "", ticker: str = "") -> str:
    """
    Determine asset class from underlying holdings, not fund features.
    
    Rules:
    - A 3x Nasdaq ETF holds equity -> 'equity' (leverage is captured by leverage_factor).
    - Cash: Money-market and Treasury funds with <1 year maturity (SHV, BIL, SGOV).
    - Fixed Income: All other bonds / debt (>1 year maturity).
    - Crypto / Digital Assets: Classified as 'alternative' (documented in schemas).
    - Real Estate: REITs / property funds -> 'real_estate'.
    - Commodities: Physical / futures commodities -> 'commodity'.
    - Equity: Large/Mid/Small cap, Sector, Thematic, International, Leveraged/Inverse Equities.
    """
    cat = (category or "").lower()
    nm = (name or "").lower()
    tick = (ticker or "").upper()

    # 1. Cash / Money Market Equivalents (<1 Year Maturity)
    if tick in {"SHV", "BIL", "SGOV"}:
        return "cash"
    if any(k in cat for k in ["money market", "ultrashort bond", "ultra short bond"]):
        return "cash"
    if any(k in nm for k in ["1-3 month", "0-1 year", "0-3 month", "t-bill"]):
        return "cash"

    # 2. Fixed Income / Bonds (>1 Year Maturity)
    if any(k in cat for k in ["bond", "fixed income", "government", "municipal", "corporate", "inflation", "tips", "high yield", "debt"]):
        return "fixed_income"
    if any(k in nm for k in ["total bond", "aggregate bond", "treasury bond", "corporate bond"]):
        return "fixed_income"

    # 3. Real Estate
    if "real estate" in cat:
        return "real_estate"
    if tick in {"VNQ", "VNQI", "SCHH", "IYR", "RWR", "REET", "XLRE"}:
        return "real_estate"
    if "real estate" in nm or re.search(r"\breit\b|\breits\b", nm):
        return "real_estate"

    # 4. Commodities
    if any(k in cat for k in ["commodit", "precious metal", "agriculture", "natural resource"]):
        return "commodity"
    if tick in {"GLD", "IAU", "SLV", "PPLT", "GDX", "GDXJ", "USO", "BNO", "DBA", "DBC", "PDBC", "GSG", "CPER"}:
        return "commodity"
    if re.search(r"\b(gold|silver|crude oil|platinum|copper|commodity|commodities)\b", nm):
        return "commodity"

    # 5. Crypto / Digital Assets -> 'alternative'
    if any(k in cat for k in ["digital asset", "crypto"]):
        return "alternative"
    if tick in {"BITO", "IBIT", "FBTC", "ETHE"}:
        return "alternative"
    if re.search(r"\b(crypto|bitcoin|ethereum)\b", nm) and not any(k in nm for k in ["equity", "fintech", "thematic", "innovation", "blockchain"]):
        return "alternative"

    # 6. Default to Equities (Broad market, sectors, themes, international, leveraged/inverse equities)
    return "equity"


# ---------------------------------------------------------------------------
# 3. RISK METRICS CALCULATOR (VOL-5band-v1 & EU SRRI)
# ---------------------------------------------------------------------------
def calc_risk_metrics(prices: pd.Series) -> Tuple[float, float, float, int, Optional[str]]:
    """
    Calculate annualised volatility (%), maximum drawdown (%), history length (years),
    and official EU SRRI (1-7) from adjusted close daily prices.
    
    Returns:
        (volatility_pct, max_drawdown_pct, history_years, srri, rejection_reason)
    """
    if prices is None or len(prices.dropna()) < 30:
        return 0.0, 0.0, 0.0, 0, "No price data or fewer than 30 points"

    clean_prices = prices.dropna()
    history_years = round(len(clean_prices) / 252.0, 2)
    
    # Requirement: Minimum 3 years of daily trading history
    if history_years < 3.0:
        return 0.0, 0.0, history_years, 0, f"Insufficient price history ({history_years} yrs < 3.0 yrs required)"

    # Daily returns (auto_adjusted closes drop false dividend/split jumps)
    daily_returns = clean_prices.pct_change().dropna()
    annual_vol = float(daily_returns.std() * math.sqrt(252) * 100.0)

    # Maximum drawdown percentage (0-100%)
    cumulative = (1.0 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min() * -100.0)

    # Range sanity checks
    if annual_vol <= 0.0 or annual_vol >= 200.0:
        return 0.0, 0.0, history_years, 0, f"Volatility out of valid range: {annual_vol:.1f}% (must be 0-200%)"
    if max_dd < 0.0 or max_dd > 100.0:
        return 0.0, 0.0, history_years, 0, f"Max drawdown out of valid range: {max_dd:.1f}%"

    # Official EU SRRI calculation (weekly Friday closes)
    try:
        weekly_prices = clean_prices.resample('W-FRI').last().dropna()
        if len(weekly_prices) >= 52:
            weekly_returns = weekly_prices.pct_change().dropna()
            weekly_vol = float(weekly_returns.std() * math.sqrt(52) * 100.0)
            if weekly_vol < 0.5:
                srri = 1
            elif weekly_vol < 2.0:
                srri = 2
            elif weekly_vol < 5.0:
                srri = 3
            elif weekly_vol < 10.0:
                srri = 4
            elif weekly_vol < 15.0:
                srri = 5
            elif weekly_vol < 25.0:
                srri = 6
            else:
                srri = 7
        else:
            srri = 0
    except Exception:
        srri = 0

    return round(annual_vol, 2), round(max_dd, 2), history_years, srri, None


def vol_to_risk(annual_vol: float, is_leveraged: bool = False, is_inverse: bool = False) -> int:
    """
    Map annualised volatility to 1-5 risk score using VOL-5band-v1 (SRRI-inspired).
    Any leveraged or inverse fund is forced to Risk 5 regardless of volatility.
    """
    if is_leveraged or is_inverse:
        return 5

    if annual_vol < 5.0:
        return 1
    elif annual_vol < 10.0:
        return 2
    elif annual_vol < 20.0:
        return 3
    elif annual_vol < 40.0:
        return 4
    else:
        return 5


# ═══════════════════════════════════════════════════════════════════════════
#  PRODUCT INGESTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

CATALOG_VERSION = "catalog_v2"
RISK_METHOD = "VOL-5band-v1 (SRRI-inspired)"


def _get_existing_mongo_tickers() -> set:
    """Retrieve tickers already present in MongoDB products collection for resuming."""
    try:
        from pymongo import MongoClient
        from app.core.config import MONGODB_URL, MONGODB_DB_NAME
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=3000)
        db = client[MONGODB_DB_NAME]
        existing = set(db["products"].distinct("ticker"))
        client.close()
        return existing
    except Exception:
        return set()


def ingest_products(resume: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch real ETF data from Yahoo Finance and calculate risk metrics.
    All data is validated and upserted directly into MongoDB.
    """
    import yfinance as yf

    tickers = ETF_TICKERS.copy()
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    price_as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    already_done = _get_existing_mongo_tickers() if resume else set()

    if already_done:
        print(f"📂 Resuming — {len(already_done)} tickers already in MongoDB, "
              f"{len(tickers) - len(already_done)} remaining")

    print(f"\n{'='*70}")
    print(f"📡  FETCHING REAL ETF DATA  ({len(tickers)} tickers)")
    print(f"    Run ID: {run_id} | Catalog: {CATALOG_VERSION}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Step 1: Batch download 5-year daily adjusted close prices
    # ------------------------------------------------------------------
    all_tickers_str = " ".join(tickers)

    print("⏳ Downloading 5-year daily prices (batch, auto-adjusted)...")
    try:
        price_data = yf.download(
            all_tickers_str,
            period="5y",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=True
        )
        print(f"✅ Price download complete — shape: {price_data.shape}")
    except Exception as e:
        print(f"❌ Batch download failed: {e}")
        price_data = pd.DataFrame()

    # ------------------------------------------------------------------
    # Step 2: Fetch .info metadata & calculate metrics
    # ------------------------------------------------------------------
    products: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    delisted: List[str] = []

    total = len(tickers)
    skipped = 0

    for idx, ticker in enumerate(tickers, 1):
        if ticker in already_done:
            skipped += 1
            continue

        pct = idx / total * 100
        print(f"\r  [{idx}/{total}] ({pct:.0f}%) Processing {ticker}...", end="", flush=True)

        res, reason = _fetch_and_validate_etf(ticker, price_data, run_id, price_as_of)

        if res:
            products.append(res)
            # Incremental batch upsert every 10 products
            if len(products) % 10 == 0:
                _save_products_to_mongo(products[-10:])
        else:
            rejected.append({"ticker": ticker, "reason": reason})
            delisted.append(ticker)

        # Rate limit pause
        time.sleep(1.2)

    # Mark disappeared / delisted tickers as active=False in MongoDB
    if delisted:
        _mark_inactive_in_mongo(delisted)

    # Final upsert for remaining products
    if products:
        _save_products_to_mongo(products)

    print(f"\n\n{'='*70}")
    print(f"✅  PRODUCT FETCH COMPLETE")
    print(f"   Total attempted: {total}")
    print(f"   Successful:      {len(products)}")
    print(f"   Rejected:        {len(rejected)}")
    if skipped:
        print(f"   Resumed/skipped: {skipped}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Step 3: Print summary & spot checks
    # ------------------------------------------------------------------
    _print_detailed_summary(products, rejected)

    return products


def _fetch_and_validate_etf(
    ticker: str, 
    price_data: pd.DataFrame, 
    run_id: str, 
    price_as_of: str
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Fetch metadata, calculate metrics, and apply range validations."""
    import yfinance as yf

    max_retries = 3
    info = None

    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            break
        except Exception as e:
            wait = (attempt + 1) * 3
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                return None, f"Failed yfinance info fetch after {max_retries} attempts ({e})"

    if not info or (info.get("regularMarketPrice") is None and info.get("previousClose") is None and info.get("navPrice") is None):
        return None, "No market price data returned (symbol may be delisted)"

    # Extract price history for ticker
    closes = _extract_closes(ticker, price_data)
    if closes is None or len(closes) == 0:
        # Fallback: single ticker download if batch omitted it
        try:
            single = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
            closes = single["Close"].dropna() if "Close" in single else None
        except Exception:
            closes = None

    # Calculate volatility, drawdown, track record, and SRRI
    vol_pct, max_dd_pct, history_yrs, srri, rej_reason = calc_risk_metrics(closes)
    if rej_reason:
        return None, rej_reason

    # Extract metadata fields
    name = info.get("longName") or info.get("shortName") or ticker
    category = info.get("category") or ""
    fund_family = info.get("fundFamily") or ""
    raw_desc = (info.get("longBusinessSummary") or "").strip()

    # Share price (current price)
    share_price = float(info.get("regularMarketPrice") or info.get("previousClose") or info.get("navPrice") or 0.0)
    share_price = round(share_price, 2)
    if share_price <= 0.0:
        return None, "Share price must be > $0.00"

    # Expense ratio check (0% to 3%)
    raw_exp = float(info.get("netExpenseRatio") or info.get("annualReportExpenseRatio") or 0.0)
    if 0.005 <= raw_exp <= 3.0:
        exp_pct = round(raw_exp, 4)  # already in percentage points (e.g. 0.0945 = 0.0945%, 0.78 = 0.78%)
    elif 0.0 < raw_exp < 0.005:
        exp_pct = round(raw_exp * 100.0, 4)  # decimal form (e.g. 0.0009 -> 0.09%)
    elif raw_exp > 3.0:
        # Check if expressed in basis points (e.g. 95 bps = 0.95%)
        if raw_exp <= 300.0:
            exp_pct = round(raw_exp / 100.0, 4)
        else:
            exp_pct = round(raw_exp, 4)
    else:
        exp_pct = 0.0

    if exp_pct < 0.0 or exp_pct > 3.0:
        return None, f"Expense ratio {exp_pct:.2f}% out of valid range (0.0% to 3.0%)"

    # 1. Leverage detection
    is_leveraged, is_inverse, leverage_factor = detect_leverage(category, name, ticker)

    # 2. Holdings-based asset class
    asset_class = category_to_asset_class(category, name, ticker)

    # 2b. Futures-based commodity detection (from Yahoo category & description, never ticker lists)
    cat_l = (category or "").lower()
    desc_l = (raw_desc or "").lower()
    name_l = (name or "").lower()
    physical_markers = ("physical gold", "physical silver", "physical platinum", "physical palladium",
                        "allocated bullion", "gold bullion", "silver bullion", "stored in vaults",
                        "holds gold bars", "holds silver bars")
    futures_markers = ("futures", "contango", "roll yield", "commodity pool", "commodities focused",
                       "oil fund", "natural gas", "forward contracts", "swap contracts")
    if any(pm in desc_l or pm in name_l for pm in physical_markers):
        is_futures_based = False
    elif asset_class == "commodity" and (
        "commodities" in cat_l or "futures" in desc_l or any(fm in desc_l or fm in name_l for fm in futures_markers)
    ):
        is_futures_based = True
    else:
        is_futures_based = any(fm in desc_l for fm in ("futures contract", "contango", "roll yield", "commodity pool"))

    # 3. Risk score calculation
    risk_score = vol_to_risk(vol_pct, is_leveraged, is_inverse)

    # 4. Fallback description synthesis
    if len(raw_desc) < 20:
        lev_str = f" with {leverage_factor}x leverage" if (is_leveraged or is_inverse) else ""
        description = (
            f"{name} ({ticker}) is an {asset_class} ETF{lev_str} in the '{category}' category, "
            f"exhibiting an annualized volatility of {vol_pct:.1f}% and a maximum drawdown of {max_dd_pct:.1f}%."
        )
    else:
        description = raw_desc[:2000]

    # Build validated document
    product_doc = {
        "ticker":            ticker,
        "name":              name,
        "risk_score":        risk_score,
        "asset_class":       asset_class,
        "share_price":       share_price,
        "price_as_of":       price_as_of,
        "description":       description,
        "expense_ratio_pct": exp_pct,
        "volatility_pct":    vol_pct,
        "max_drawdown_pct":  max_dd_pct,
        "fund_family":       fund_family,
        "is_leveraged":      is_leveraged,
        "is_inverse":        is_inverse,
        "leverage_factor":   leverage_factor,
        "is_futures_based":  is_futures_based,
        "history_years":     history_yrs,
        "srri":              srri if srri > 0 else None,
        "catalog_version":   CATALOG_VERSION,
        "active":            True,
        # Backward-compatibility alias
        "min_investment":    share_price,
        # Audit & provenance metadata
        "_run_id":           run_id,
        "_category":         category,
        "_fetched_at":       datetime.now(timezone.utc).isoformat(),
        "_data_source":      "yfinance",
        "_risk_method":      RISK_METHOD,
    }

    return product_doc, ""


def _extract_closes(ticker: str, price_data: pd.DataFrame) -> Optional[pd.Series]:
    """Safely extract Close series for a ticker from a multi-index or single DataFrame."""
    try:
        if isinstance(price_data.columns, pd.MultiIndex):
            if ticker in price_data.columns.get_level_values(0):
                return price_data[ticker]["Close"].dropna()
        else:
            if "Close" in price_data:
                return price_data["Close"].dropna()
    except Exception:
        pass
    return None


def _save_products_to_mongo(products: List[Dict[str, Any]]):
    """Upsert products to MongoDB collection."""
    try:
        from pymongo import MongoClient, ReplaceOne
        from app.core.config import MONGODB_URL, MONGODB_DB_NAME

        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
        db = client[MONGODB_DB_NAME]
        col = db["products"]
        col.create_index("ticker", unique=True)

        ops = [ReplaceOne({"ticker": d["ticker"]}, d, upsert=True) for d in products]
        result = col.bulk_write(ops, ordered=False)
        client.close()
    except Exception as e:
        print(f"\n⚠️  MongoDB save failed: {e}")


def _mark_inactive_in_mongo(delisted_tickers: List[str]):
    """Mark delisted or missing tickers as active: False in MongoDB."""
    try:
        from pymongo import MongoClient
        from app.core.config import MONGODB_URL, MONGODB_DB_NAME

        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=4000)
        db = client[MONGODB_DB_NAME]
        res = db["products"].update_many(
            {"ticker": {"$in": delisted_tickers}},
            {"$set": {"active": False, "_delisted_at": datetime.now(timezone.utc).isoformat()}}
        )
        if res.modified_count > 0:
            print(f"ℹ️  Marked {res.modified_count} delisted products as active=False in MongoDB")
        client.close()
    except Exception:
        pass


def _print_detailed_summary(products: List[Dict[str, Any]], rejected: List[Dict[str, Any]]):
    """Print comprehensive post-ingestion summary, spot-checks, and distribution comparison."""
    risk_labels = {1: "Conservative", 2: "Mod-Conservative", 3: "Moderate",
                   4: "Mod-Aggressive", 5: "Aggressive"}

    risk_counts = {}
    ac_counts = {}
    srri_counts = {}

    for p in products:
        r = p["risk_score"]
        a = p["asset_class"]
        s = p.get("srri") or 0
        risk_counts[r] = risk_counts.get(r, 0) + 1
        ac_counts[a] = ac_counts.get(a, 0) + 1
        srri_counts[s] = srri_counts.get(s, 0) + 1

    print("=" * 70)
    print("📊 COMPREHENSIVE INGESTION SUMMARY")
    print("=" * 70)
    print(f"Total Attempted: {len(products) + len(rejected)}")
    print(f"Total Ingested:  {len(products)}")
    print(f"Total Rejected:  {len(rejected)}")
    
    if rejected:
        print("\n❌ Rejected Products Breakdown:")
        for r in rejected:
            print(f"   • {r['ticker']:6s}: {r['reason']}")

    print("\n📈 Risk Level Distribution (1-5 Scale: VOL-5band-v1):")
    for r in sorted(risk_counts.keys()):
        pct = risk_counts[r] / len(products) * 100
        bar = "█" * int(pct // 2)
        print(f"   Risk {r} ({risk_labels.get(r, ''):17s}): {risk_counts[r]:3d} ({pct:5.1f}%) {bar}")

    print("\n🏛️ Asset Class Distribution (Holdings-Based):")
    for a in sorted(ac_counts.keys()):
        pct = ac_counts[a] / len(products) * 100
        bar = "█" * int(pct // 2)
        print(f"   {a:14s}: {ac_counts[a]:3d} ({pct:5.1f}%) {bar}")

    print("\n🇪🇺 Official EU SRRI (1-7 Scale: Annualised Volatility from Weekly Returns) vs VOL-5band-v1:")
    print("   SRRI Level | Annualised Vol Band | Count | Share  | CESR/ESMA Category")
    print("   " + "-" * 72)
    srri_bands = {
        1: "< 0.5%",
        2: "0.5% - 2.0%",
        3: "2.0% - 5.0%",
        4: "5.0% - 10.0%",
        5: "10.0% - 15.0%",
        6: "15.0% - 25.0%",
        7: ">= 25.0%",
    }
    for s in range(1, 8):
        c = srri_counts.get(s, 0)
        pct = c / len(products) * 100 if len(products) else 0
        band = srri_bands.get(s, "")
        print(f"   SRRI {s}     | {band:19s} | {c:5d} | {pct:5.1f}% | European UCITS band {s}")

    # Spot checks
    print("\n🔍 Spot-Check Verification on Benchmark Funds:")
    print("   Ticker | Name                          | Risk | Asset Class  | Lev | Inv | Volatility")
    print("   " + "-" * 75)
    lookup = {p["ticker"]: p for p in products}
    benchmarks = ["SHV", "BIL", "BND", "AGG", "SPY", "QQQ", "TQQQ", "SQQQ", "BITO"]
    for b in benchmarks:
        if b in lookup:
            p = lookup[b]
            lev_str = f"{p['leverage_factor']:+.0f}x"
            inv_str = "Yes" if p["is_inverse"] else "No"
            print(f"   {b:6s} | {p['name'][:30]:30s} | {p['risk_score']:4d} | {p['asset_class']:12s} | {lev_str:3s} | {inv_str:3s} | {p['volatility_pct']:6.2f}%")
        else:
            print(f"   {b:6s} | NOT FOUND IN INGESTED CATALOG")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  INVESTOR GENERATION (delegated to scripts/generate_investors.py — synthetic-v2)
# ═══════════════════════════════════════════════════════════════════════════

def run_investor_generation(count: int = 5500):
    """Generate synthetic-v2 investor profiles via the single canonical generator."""
    from scripts.generate_investors import generate_investors, print_summary, seed_mongodb
    profiles = generate_investors(total=count)
    print_summary(profiles)
    seed_mongodb(profiles)


# ═══════════════════════════════════════════════════════════════════════════
#  FAISS INDEX REBUILD
# ═══════════════════════════════════════════════════════════════════════════

def rebuild_faiss_index():
    """
    Rebuild the FAISS vector store from MongoDB products collection.
    Tracks embedding model provenance to prevent mixing vector spaces.
    """
    from app.schemas.schemas import Product
    from app.services.vector_store import VectorStore, ProductDocument
    from app.services.jina_embedder import get_embedder
    from app.core.database import DatabaseManager

    print(f"\n{'='*70}")
    print(f"🏗️  REBUILDING FAISS VECTOR INDEX")
    print(f"{'='*70}\n")

    embedder = get_embedder(use_real=True)
    model_name = getattr(embedder, "prime_model", "jina-embeddings-v3")

    # Load products from MongoDB
    print("📦 Loading products from MongoDB...")
    db = DatabaseManager()
    if not db.is_available():
        raise RuntimeError("MongoDB is required but not connected. Please ensure MongoDB is running.")
    products = db.get_all_products(active_only=True)
    db.close()

    if not products:
        raise RuntimeError("No active products found in MongoDB. Run 'python3 -m scripts.ingest_data --products' first.")

    print(f"✅ Loaded {len(products)} active products from MongoDB")

    # Verify 1-to-1 mapping: every FAISS document must map to exactly one active ticker
    seen_tickers = set()
    for product in products:
        if product.ticker in seen_tickers:
            raise ValueError(f"Duplicate ticker in MongoDB active products: {product.ticker}")
        seen_tickers.add(product.ticker)
    print(f"✅ Verified 1-to-1 mapping: exactly {len(seen_tickers)} unique active tickers to be indexed")

    # Build rich descriptions for embedding
    descriptions = []
    product_docs = []

    for product in products:
        desc = product.description or ""
        if len(desc) < 20:
            lev_str = f" with {product.leverage_factor}x leverage" if (product.is_leveraged or product.is_inverse) else ""
            desc = (
                f"{product.name} ({product.ticker}) is an {product.asset_class} ETF{lev_str}, "
                f"risk {product.risk_score}/5, annual volatility {product.volatility_pct:.1f}%."
            )
        else:
            lev_str = f", {product.leverage_factor:+.0f}x leverage" if (product.is_leveraged or product.is_inverse) else ""
            desc = (
                f"{product.name} ({product.ticker}) — "
                f"{product.asset_class}{lev_str}, "
                f"risk {product.risk_score}/5. {desc}"
            )

        descriptions.append(desc)
        product_docs.append(ProductDocument(
            ticker=product.ticker,
            name=product.name,
            description=desc,
            risk_score=product.risk_score,
            asset_class=product.asset_class,
            min_investment=product.share_price,
            share_price=product.share_price,
            metadata={
                "risk_score":        product.risk_score,
                "asset_class":       product.asset_class,
                "volatility_pct":    product.volatility_pct,
                "fund_family":       product.fund_family,
                "is_leveraged":      product.is_leveraged,
                "is_inverse":        product.is_inverse,
                "leverage_factor":   product.leverage_factor,
                "srri":              product.srri,
                "embedding_model":   model_name,
            }
        ))

    print(f"📝 Built {len(descriptions)} descriptions for embedding")

    # Embed in batches (Jina API has batch limits)
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        print(f"   Embedding batch {i // batch_size + 1}/{(len(descriptions) - 1) // batch_size + 1} "
              f"({len(batch)} docs)...")
        emb = embedder.embed_documents(batch)
        all_embeddings.append(emb)
        if i + batch_size < len(descriptions):
            time.sleep(0.5)

    embeddings = np.vstack(all_embeddings)
    print(f"✅ Generated embeddings — shape: {embeddings.shape} (Model: {model_name})")

    # Build FAISS index and save with provenance
    vector_store = VectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add_documents(product_docs, embeddings)

    os.makedirs("data", exist_ok=True)
    vector_store.save("data/product_vector_store", model_name=model_name)

    print(f"\n{'='*70}")
    print(f"✅  VECTOR STORE REBUILD COMPLETE!")
    print(f"   Products indexed: {len(product_docs)}")
    print(f"   Embedding Model:  {model_name}")
    print(f"   Embedding Dim:    {embeddings.shape[1]}")
    print(f"   Files:")
    print(f"     → data/product_vector_store.index")
    print(f"     → data/product_vector_store.docs")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ethical AI Investment Advisor — Data Ingestion Pipeline"
    )
    parser.add_argument("--products", action="store_true",
                        help="Fetch real ETF products only")
    parser.add_argument("--investors", action="store_true",
                        help="Generate synthetic investors only")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild FAISS vector index only")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted product fetch")
    parser.add_argument("--investor-count", type=int, default=5500,
                        help="Number of synthetic investors (default: 5500)")

    args = parser.parse_args()
    run_all = not (args.products or args.investors or args.rebuild)

    start = time.time()
    print(f"\n{'═'*70}")
    print(f"  ETHICAL AI INVESTMENT ADVISOR — DATA INGESTION PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*70}\n")

    if run_all or args.products:
        ingest_products(resume=args.resume)

    if run_all or args.investors:
        run_investor_generation(count=args.investor_count)

    if run_all or args.rebuild:
        rebuild_faiss_index()

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{'═'*70}")
    print(f"  🎉  INGESTION COMPLETE — Total time: {minutes}m {seconds}s")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
