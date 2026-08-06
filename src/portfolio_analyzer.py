"""
API-driven Portfolio Holdings Analyzer.

Purpose:
- fetch ticker info automatically
- classify holdings into broad asset classes
- support fund look-through when API data is available
- roll up sector-style outputs into Monte Carlo-friendly buckets
- fall back cleanly when classification is uncertain

This is designed to feed a regime-aware Monte Carlo engine.

Design goals:
- correctness over aggressiveness
- do not misclassify international funds as US equity
- avoid blind trust in Yahoo fund payloads
- prefer fund identity over raw holdings if identity is obvious
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import yfinance as yf


@dataclass
class Holding:
    ticker: str
    shares: float
    current_price: float = 0.0
    market_value: float = 0.0
    weight: float = 0.0
    asset_class: str = "Unknown"
    raw_exposure: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class TickerClassifier:
    """
    Classifies securities using live API metadata and broad asset-class rollups.
    """

    BROAD_CLASSES = {
        "US_Equity",
        "International_Equity",
        "Emerging_Markets_Equity",
        "US_Bonds",
        "Real_Estate",
        "Commodities",
        "Cash",
        "Unknown",
    }

    ROLLUP_MAP = {
        # Equity sleeves
        "US_Large_Cap": "US_Equity",
        "US_Mid_Cap": "US_Equity",
        "US_Small_Cap": "US_Equity",
        "US_Equity": "US_Equity",
        "Developed_International_Equity": "International_Equity",
        "International_Equity": "International_Equity",
        "Emerging_Markets_Equity": "Emerging_Markets_Equity",

        # Bond sleeves
        "Core_Bonds": "US_Bonds",
        "Municipal_Bonds": "US_Bonds",
        "High_Yield_Bonds": "US_Bonds",
        "Global_Bonds": "US_Bonds",
        "Inflation_Linked_Bonds": "US_Bonds",
        "US_Bonds": "US_Bonds",

        # Other
        "Real_Estate": "Real_Estate",
        "Commodities": "Commodities",
        "Cash": "Cash",
        "Unknown": "Unknown",
    }

    POSITION_FIELD_TO_ASSET_CLASS = {
        "stockposition": "US_Equity",
        "bondposition": "US_Bonds",
        "preferredposition": "US_Bonds",
        "convertibleposition": "US_Bonds",
        # "otherposition" should not become Unknown if there is real cash in the fund
        # but it also should not become equity. Conservative default:
        "otherposition": "Cash",
        "cashposition": "Cash",
    }

    SECTOR_TO_ASSET_CLASS = {
        "technology": "US_Equity",
        "industrials": "US_Equity",
        "financial_services": "US_Equity",
        "consumer_cyclical": "US_Equity",
        "consumer_defensive": "US_Equity",
        "healthcare": "US_Equity",
        "energy": "US_Equity",
        "basic_materials": "US_Equity",
        "utilities": "US_Equity",
        "communication_services": "US_Equity",
        "realestate": "Real_Estate",
        "real_estate": "Real_Estate",
        "materials": "US_Equity",
        "financials": "US_Equity",
        "consumer_goods": "US_Equity",
    }

    CATEGORY_TO_ASSET_CLASS = {
        # US equity
        "large blend": "US_Equity",
        "large growth": "US_Equity",
        "large value": "US_Equity",
        "mid blend": "US_Equity",
        "mid growth": "US_Equity",
        "mid value": "US_Equity",
        "small blend": "US_Equity",
        "small growth": "US_Equity",
        "small value": "US_Equity",

        # international equity
        "foreign large blend": "International_Equity",
        "foreign large growth": "International_Equity",
        "foreign large value": "International_Equity",
        "foreign small/mid blend": "International_Equity",
        "foreign small/mid growth": "International_Equity",
        "foreign small/mid value": "International_Equity",
        "international": "International_Equity",
        "international equity": "International_Equity",
        "developed markets": "International_Equity",

        # emerging markets
        "diversified emerging mkts": "Emerging_Markets_Equity",
        "emerging markets": "Emerging_Markets_Equity",

        # bonds
        "intermediate core bond": "US_Bonds",
        "short government": "US_Bonds",
        "ultrashort bond": "Cash",
        "corporate bond": "US_Bonds",
        "high yield bond": "US_Bonds",
        "long government": "US_Bonds",
        "short-term bond": "US_Bonds",
        "inflation-protected bond": "US_Bonds",
        "global bond-usd hedged": "US_Bonds",

        # real assets
        "real estate": "Real_Estate",
        "commodities focused": "Commodities",
        "commodities broad basket": "Commodities",
    }

    BALANCED_CATEGORY_HINTS = [
        "balanced",
        "allocation",
        "moderate allocation",
        "conservative allocation",
        "aggressive allocation",
        "income allocation",
        "target allocation",
        "moderately conservative",
        "moderately aggressive",
        "lifecycle",
        "retirement",
        "80/20",
        "60/40",
        "70/30",
    ]

    INTERNATIONAL_HINTS = [
        "international",
        "foreign",
        "ex us",
        "ex-us",
        "non-us",
        "non us",
        "developed ex us",
        "total international",
        "world ex us",
        "global ex us",
        "foreign large blend",
        "foreign large growth",
        "foreign large value",
        "foreign small/mid blend",
        "foreign small/mid growth",
        "foreign small/mid value",
        "international equity",
        "developed markets",
    ]

    # High-confidence ticker hints for common ETFs / mutual funds
    TICKER_HINTS = {
        "VXUS": {"International_Equity": 1.0},
        "VEA": {"International_Equity": 1.0},
        "VWO": {"Emerging_Markets_Equity": 1.0},
        "IXUS": {"International_Equity": 1.0},
        "VEU": {"International_Equity": 1.0},
        "VOO": {"US_Equity": 1.0},
        "VTI": {"US_Equity": 1.0},
        "SPY": {"US_Equity": 1.0},
        "IVV": {"US_Equity": 1.0},
        "BND": {"US_Bonds": 1.0},
        "AGG": {"US_Bonds": 1.0},
        "SCHZ": {"US_Bonds": 1.0},
        "GOVT": {"US_Bonds": 1.0},
        "VNQ": {"Real_Estate": 1.0},
        "IYR": {"Real_Estate": 1.0},
        "SCHH": {"Real_Estate": 1.0},
        # Futures-based commodity ETFs: `asset_classes` look-through reports these
        # as mostly Cash (collateral backing the futures), which is technically
        # true but economically misleading -- their actual market exposure is
        # commodities. Hard-hinted here rather than relying on the cash-sanity-check
        # fallback, since these are well-known and unambiguous.
        "PDBC": {"Commodities": 1.0},
        "DBC": {"Commodities": 1.0},
        "GSG": {"Commodities": 1.0},
        "USO": {"Commodities": 1.0},
    }

    def __init__(self, manual_overrides_path: str = "config/manual_fund_overrides.json"):
        self.manual_overrides = self._load_manual_overrides(manual_overrides_path)

    # -------------------------------------------------------------------------
    # Loading / data access
    # -------------------------------------------------------------------------
    def _load_manual_overrides(self, path_str: str) -> Dict[str, Dict[str, float]]:
        path = Path(path_str)
        if not path.exists():
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        return {k.upper(): v for k, v in data.items()}

    def fetch_metadata(self, ticker: str) -> Dict[str, Any]:
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}

        return {
            "ticker": ticker.upper(),
            "quoteType": str(info.get("quoteType", "")).lower(),
            "category": str(info.get("category", "")).lower(),
            "fundFamily": str(info.get("fundFamily", "")).lower(),
            "sector": str(info.get("sector", "")).lower(),
            "industry": str(info.get("industry", "")).lower(),
            "longName": str(info.get("longName", "")).lower(),
            "shortName": str(info.get("shortName", "")).lower(),
        }

    def fetch_price(self, ticker: str) -> float:
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        if price is None:
            hist = t.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        if price is None:
            raise ValueError(f"Could not fetch price for {ticker}")

        return float(price)

    # -------------------------------------------------------------------------
    # Parsing / normalization
    # -------------------------------------------------------------------------
    def _parse_holdings_payload(self, payload: Any) -> Optional[Dict[str, float]]:
        if isinstance(payload, dict):
            out = {}
            for k, v in payload.items():
                try:
                    weight = float(v)
                    if weight > 1.0:
                        weight = weight / 100.0
                    weight = max(weight, 0.0)
                    out[str(k)] = weight
                except Exception:
                    continue
            return out or None

        if isinstance(payload, list):
            out = {}
            for item in payload:
                if not isinstance(item, dict):
                    continue

                name = (
                    item.get("name")
                    or item.get("symbol")
                    or item.get("holding")
                    or item.get("assetClass")
                    or item.get("sector")
                    or item.get("category")
                )
                weight = (
                    item.get("percent")
                    or item.get("weight")
                    or item.get("allocation")
                    or item.get("ytdReturn")
                )

                if name is None or weight is None:
                    continue

                try:
                    w = float(weight)
                    if w > 1.0:
                        w = w / 100.0
                    w = max(w, 0.0)
                    out[str(name)] = w
                except Exception:
                    continue

            return out or None

        return None

    def _normalize_sleeve_name(self, name: str) -> str:
        s = str(name).lower().strip()

        if s in self.SECTOR_TO_ASSET_CLASS:
            return s

        if s in self.POSITION_FIELD_TO_ASSET_CLASS:
            return s

        # equity sleeves
        if any(x in s for x in ["large", "large cap", "us large"]):
            return "US_Large_Cap"
        if any(x in s for x in ["mid", "mid cap"]):
            return "US_Mid_Cap"
        if any(x in s for x in ["small", "small cap"]):
            return "US_Small_Cap"
        if any(x in s for x in ["emerging", "emerg"]):
            return "Emerging_Markets_Equity"
        if any(x in s for x in ["international", "foreign", "developed", "ex us", "ex-us", "non-us", "non us"]):
            return "Developed_International_Equity"

        # bond sleeves
        if any(x in s for x in ["core fixed", "core bond", "aggregate", "intermediate", "investment grade"]):
            return "Core_Bonds"
        if "muni" in s or "municipal" in s:
            return "Municipal_Bonds"
        if any(x in s for x in ["high yield", "junk"]):
            return "High_Yield_Bonds"
        if "global bond" in s or "world bond" in s:
            return "Global_Bonds"
        if "tips" in s or "inflation" in s:
            return "Inflation_Linked_Bonds"

        # alternatives
        if any(x in s for x in ["commodity", "gold", "silver", "oil", "natural resources"]):
            return "Commodities"
        if any(x in s for x in ["reit", "real estate"]):
            return "Real_Estate"
        if any(x in s for x in ["cash", "money market", "liquid"]):
            return "Cash"

        return name

    def _normalize_and_rollup_exposure(
        self,
        exposure: Dict[str, float],
        *,
        international_context: bool = False,
    ) -> Dict[str, float]:
        """
        Normalize raw exposures and roll them into broad simulation buckets.
        """
        broad: Dict[str, float] = {}

        for raw_class, weight in exposure.items():
            norm = self._normalize_sleeve_name(raw_class)
            lower_norm = str(norm).lower().strip()

            if lower_norm in self.POSITION_FIELD_TO_ASSET_CLASS:
                rolled = self.POSITION_FIELD_TO_ASSET_CLASS[lower_norm]
            elif lower_norm in self.SECTOR_TO_ASSET_CLASS:
                rolled = self.SECTOR_TO_ASSET_CLASS[lower_norm]
            else:
                rolled = self.ROLLUP_MAP.get(norm, self.ROLLUP_MAP.get(lower_norm, norm))

            if international_context and rolled == "US_Equity":
                rolled = "International_Equity"

            broad[rolled] = broad.get(rolled, 0.0) + float(weight)

        total = sum(broad.values())
        if total > 0:
            broad = {k: v / total for k, v in broad.items()}

        return broad or {"Unknown": 1.0}

    # -------------------------------------------------------------------------
    # Heuristics
    # -------------------------------------------------------------------------
    def _infer_balanced_exposure_from_text(self, text: str) -> Optional[Dict[str, float]]:
        if not any(hint in text for hint in self.BALANCED_CATEGORY_HINTS):
            return None

        # Conservative defaults
        if "conservative" in text or "income" in text:
            return {"US_Equity": 0.60, "US_Bonds": 0.40}
        if "aggressive" in text:
            return {"US_Equity": 0.85, "US_Bonds": 0.15}
        if "moderate" in text:
            return {"US_Equity": 0.70, "US_Bonds": 0.30}

        return {"US_Equity": 0.80, "US_Bonds": 0.20}

    def _is_international_fund(self, ticker: str, metadata: Dict[str, Any]) -> bool:
        ticker_upper = ticker.upper()
        if ticker_upper in {"VXUS", "VEA", "VWO", "IXUS", "VEU"}:
            return True

        text = " ".join([
            metadata.get("quoteType", "") or "",
            metadata.get("category", "") or "",
            metadata.get("fundFamily", "") or "",
            metadata.get("sector", "") or "",
            metadata.get("industry", "") or "",
            metadata.get("longName", "") or "",
            metadata.get("shortName", "") or "",
        ]).lower()

        return any(hint in text for hint in self.INTERNATIONAL_HINTS)

    def _is_balanced_fund(self, metadata: Dict[str, Any]) -> bool:
        text = " ".join([
            metadata.get("quoteType", "") or "",
            metadata.get("category", "") or "",
            metadata.get("fundFamily", "") or "",
            metadata.get("sector", "") or "",
            metadata.get("industry", "") or "",
            metadata.get("longName", "") or "",
            metadata.get("shortName", "") or "",
        ]).lower()

        if any(hint in text for hint in self.BALANCED_CATEGORY_HINTS):
            return True

        return False

    # -------------------------------------------------------------------------
    # Fund holdings fetch / ranking
    # -------------------------------------------------------------------------
    def try_fetch_fund_holdings(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Try to fetch allocation / holdings-style data from Yahoo.
        Prefer asset-class allocations over sector weightings.
        """
        t = yf.Ticker(ticker)
        candidate_payloads: List[Tuple[str, Any]] = []

        try:
            info = t.info or {}
        except Exception:
            info = {}

        for key in ["assetClass", "holdings", "topHoldings", "sectorWeightings", "fundHoldings"]:
            if key in info and info[key] is not None:
                candidate_payloads.append((f"info.{key}", info[key]))

        for attr in ["funds_data"]:
            obj = getattr(t, attr, None)
            if obj is not None:
                for field in ["asset_classes", "equity_holdings", "top_holdings", "sector_weightings"]:
                    try:
                        value = getattr(obj, field, None)
                        if value is not None:
                            candidate_payloads.append((f"{attr}.{field}", value))
                    except Exception:
                        continue

        parsed_payloads: List[Tuple[str, Dict[str, float]]] = []
        for src, payload in candidate_payloads:
            parsed = self._parse_holdings_payload(payload)
            if parsed:
                parsed_payloads.append((src, parsed))

        if not parsed_payloads:
            return None

        def score_source(src: str, parsed: Dict[str, float]) -> float:
            src_lower = src.lower()
            broad = self._normalize_and_rollup_exposure(parsed)

            score = 0.0

            is_asset_classes_src = "asset_classes" in src_lower or "assetclass" in src_lower
            is_sector_src = "sector_weightings" in src_lower

            if is_asset_classes_src:
                score += 100.0

            if "top_holdings" in src_lower or "holdings" in src_lower:
                score += 40.0

            if is_sector_src:
                score += 5.0

            if "US_Bonds" in broad:
                score += 20.0

            if len(broad) > 1:
                score += 10.0

            if broad.get("US_Equity", 0.0) > 0.9:
                score -= 5.0

            # An `asset_classes` payload (stock/bond/cash split) only tells us how
            # much of the fund is equity-like -- it says nothing about WHAT that
            # equity actually is. For single-sector funds (REITs, sector ETFs,
            # commodity-equity funds) this payload looks like "98% US_Equity" even
            # though the underlying holdings are 99% real estate. If a sector-level
            # payload for this SAME ticker clearly identifies a concentrated
            # non-equity sector (real estate, commodities), that's strictly more
            # informative and should win regardless of the generic source-name bonus.
            if is_asset_classes_src:
                dominant_asset_class = max(broad, key=broad.get) if broad else None
                if dominant_asset_class == "US_Equity" and broad.get("US_Equity", 0.0) > 0.85:
                    score -= 90.0

            if is_sector_src:
                dominant_sector_class = max(broad, key=broad.get) if broad else None
                if dominant_sector_class in ("Real_Estate", "Commodities") and broad.get(dominant_sector_class, 0.0) > 0.5:
                    score += 90.0

            return score

        parsed_payloads.sort(key=lambda x: score_source(x[0], x[1]), reverse=True)
        _, best_payload = parsed_payloads[0]
        return best_payload

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------
    def classify_metadata(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        q = metadata.get("quoteType", "")
        cat = metadata.get("category", "")
        sector = metadata.get("sector", "")
        industry = metadata.get("industry", "")
        name = metadata.get("longName", "") + " " + metadata.get("shortName", "")

        text = f"{q} {cat} {sector} {industry} {name}".lower()

        if q == "cryptocurrency":
            return {"Unknown": 1.0}

        if any(x in text for x in ["money market", "cash", "treasury bill", "short treasury"]):
            return {"Cash": 1.0}

        if q in ["etf", "mutualfund"]:
            if cat in self.CATEGORY_TO_ASSET_CLASS:
                return {self.CATEGORY_TO_ASSET_CLASS[cat]: 1.0}

            if self._is_international_fund(metadata.get("ticker", ""), metadata):
                return {"International_Equity": 1.0}

            balanced = self._infer_balanced_exposure_from_text(text)
            if balanced is not None:
                return balanced

            if any(x in text for x in ["bond", "income", "treasury", "fixed income"]):
                return {"US_Bonds": 1.0}

            if "real estate" in text or "reit" in text:
                return {"Real_Estate": 1.0}

            if any(x in text for x in ["commodity", "gold", "silver", "oil", "natural resources"]):
                return {"Commodities": 1.0}

            return {"Unknown": 1.0}

        if any(x in text for x in ["bond", "fixed income", "income", "treasury", "municipal", "high yield"]):
            if "municipal" in text:
                return {"Municipal_Bonds": 1.0}
            if "high yield" in text:
                return {"High_Yield_Bonds": 1.0}
            if "global" in text or "international" in text:
                return {"Global_Bonds": 1.0}
            if "inflation" in text or "tips" in text:
                return {"Inflation_Linked_Bonds": 1.0}
            return {"Core_Bonds": 1.0}

        if any(x in text for x in ["reit", "real estate"]):
            return {"Real_Estate": 1.0}
        if any(x in text for x in ["commodity", "gold", "silver", "oil", "natural resources"]):
            return {"Commodities": 1.0}

        if q == "equity":
            if sector and sector in self.SECTOR_TO_ASSET_CLASS:
                return {sector: 1.0}
            return {"US_Equity": 1.0}

        return {"Unknown": 1.0}

    def classify_ticker(self, ticker: str) -> Dict[str, float]:
        """
        Main entry point.

        Priority:
        1) explicit manual override
        2) hard ticker hint
        3) explicit international / balanced identity
        4) fund look-through
        5) metadata/category inference
        6) conservative fallback
        """
        ticker = ticker.upper()

        # 1) explicit manual override
        if ticker in self.manual_overrides:
            return self.manual_overrides[ticker].copy()

        # 2) hard ticker hint
        if ticker in self.TICKER_HINTS:
            return self.TICKER_HINTS[ticker].copy()

        metadata = self.fetch_metadata(ticker)
        q = (metadata.get("quoteType") or "").lower().strip()
        cat = (metadata.get("category") or "").lower().strip()
        text = " ".join([
            q,
            cat,
            metadata.get("fundFamily", "") or "",
            metadata.get("sector", "") or "",
            metadata.get("industry", "") or "",
            metadata.get("longName", "") or "",
            metadata.get("shortName", "") or "",
        ]).lower()

        if q in ["etf", "mutualfund"]:
            # Hard international detection first
            if self._is_international_fund(ticker, metadata):
                fund_holdings = self.try_fetch_fund_holdings(ticker)
                if fund_holdings:
                    fund_broad = self._normalize_and_rollup_exposure(
                        fund_holdings,
                        international_context=True,
                    )
                    return fund_broad or {"International_Equity": 1.0}
                return {"International_Equity": 1.0}

            # Balanced funds next
            if self._is_balanced_fund(metadata):
                fund_holdings = self.try_fetch_fund_holdings(ticker)
                if fund_holdings:
                    fund_broad = self._normalize_and_rollup_exposure(fund_holdings)
                    if fund_broad:
                        # If look-through is suspiciously one-sided and no bonds, use balanced heuristic.
                        if fund_broad.get("US_Equity", 0.0) > 0.9 and "US_Bonds" not in fund_broad:
                            balanced = self._infer_balanced_exposure_from_text(text)
                            if balanced is not None:
                                return balanced
                        return fund_broad

                balanced = self._infer_balanced_exposure_from_text(text)
                if balanced is not None:
                    return balanced

            # Fund look-through
            fund_holdings = self.try_fetch_fund_holdings(ticker)
            if fund_holdings:
                fund_broad = self._normalize_and_rollup_exposure(fund_holdings)
                if fund_broad and next(iter(fund_broad.keys())) != "Unknown":
                    # Sanity check: a look-through payload dominated by Cash is
                    # sometimes technically accurate but economically misleading --
                    # e.g. futures-based commodity ETFs (PDBC, DBC, USO) hold most
                    # assets as cash/Treasury collateral backing derivatives, so
                    # `asset_classes` legitimately reports ~85%+ cash even though the
                    # fund's actual market exposure is commodities. If the category
                    # metadata clearly identifies a different, non-cash asset class,
                    # prefer that over a cash-dominated look-through result.
                    if fund_broad.get("Cash", 0.0) > 0.70:
                        category_result = self.classify_metadata(metadata)
                        category_broad = self._normalize_and_rollup_exposure(category_result)
                        category_label = next(iter(category_broad.keys())) if category_broad else "Unknown"
                        if category_label not in ("Unknown", "Cash"):
                            return category_broad
                    return fund_broad

            # Metadata classification
            category_result = self.classify_metadata(metadata)
            category_broad = self._normalize_and_rollup_exposure(category_result)
            if category_broad and next(iter(category_broad.keys())) != "Unknown":
                return category_broad

            return {"Unknown": 1.0}

        # Non-fund securities
        fund_holdings = self.try_fetch_fund_holdings(ticker)
        if fund_holdings:
            fund_broad = self._normalize_and_rollup_exposure(fund_holdings)
            if fund_broad and next(iter(fund_broad.keys())) != "Unknown":
                return fund_broad

        raw = self.classify_metadata(metadata)
        rolled = self._normalize_and_rollup_exposure(raw)
        if rolled and next(iter(rolled.keys())) != "Unknown":
            return rolled

        return {"Unknown": 1.0}

    def rollup_to_broad_classes(self, exposure: Dict[str, float]) -> Dict[str, float]:
        return self._normalize_and_rollup_exposure(exposure)


class PortfolioAnalyzer:
    """
    Analyzes a portfolio, prices holdings, and computes broad asset-class weights.
    """

    def __init__(self):
        self.holdings: List[Holding] = []
        self.total_value: float = 0.0
        self.asset_class_weights: Dict[str, float] = {}
        self.classifier = TickerClassifier()

    def load_portfolio_from_csv(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        if "ticker" not in df.columns or "shares" not in df.columns:
            raise ValueError("CSV must contain columns: ticker,shares")

        self.holdings = [
            Holding(ticker=str(row["ticker"]).upper(), shares=float(row["shares"]))
            for _, row in df.iterrows()
        ]

    def load_portfolio_from_dict(self, portfolio: Dict[str, float]) -> None:
        self.holdings = [
            Holding(ticker=str(t).upper(), shares=float(s))
            for t, s in portfolio.items()
        ]

    def fetch_current_prices(self) -> None:
        print("[*] Fetching current prices...")

        for h in self.holdings:
            try:
                h.current_price = self.classifier.fetch_price(h.ticker)
                h.market_value = h.current_price * h.shares
                print(
                    f"    {h.ticker}: ${h.current_price:.2f} x {h.shares:.2f} = "
                    f"${h.market_value:,.2f}"
                )
            except Exception as e:
                print(f"    ⚠️  {h.ticker}: price lookup failed ({e})")
                h.current_price = 0.0
                h.market_value = 0.0

        self.total_value = sum(h.market_value for h in self.holdings)
        print(f"[+] Total portfolio value: ${self.total_value:,.2f}")

    def compute_weights(self) -> None:
        """
        Convert each holding into broad asset-class weights.
        """
        if self.total_value <= 0:
            raise ValueError("Total portfolio value is zero.")

        asset_values: Dict[str, float] = {}

        for h in self.holdings:
            if h.market_value <= 0:
                continue

            raw_exposure = self.classifier.classify_ticker(h.ticker)
            h.raw_exposure = raw_exposure.copy()

            broad_exposure = self.classifier.rollup_to_broad_classes(raw_exposure)
            h.asset_class = (
                "Multi_Asset_Fund"
                if len(broad_exposure) > 1
                else next(iter(broad_exposure.keys()))
            )

            for ac, w in broad_exposure.items():
                asset_values[ac] = asset_values.get(ac, 0.0) + h.market_value * float(w)

        self.asset_class_weights = {
            ac: val / self.total_value
            for ac, val in asset_values.items()
        }

        print("\n[+] Portfolio Asset Allocation:")
        for ac, w in sorted(self.asset_class_weights.items(), key=lambda x: -x[1]):
            print(f"    {ac}: {w:.2%}")

    def analyze(self, portfolio_input) -> Dict:
        if isinstance(portfolio_input, str):
            self.load_portfolio_from_csv(portfolio_input)
        elif isinstance(portfolio_input, dict):
            self.load_portfolio_from_dict(portfolio_input)
        else:
            raise ValueError("portfolio_input must be a CSV path or a dict")

        self.fetch_current_prices()
        self.compute_weights()
        return self.get_summary()

    def get_summary(self) -> Dict:
        return {
            "total_value": self.total_value,
            "num_holdings": len(self.holdings),
            "asset_class_weights": self.asset_class_weights,
            "holdings": [
                {
                    "ticker": h.ticker,
                    "shares": h.shares,
                    "price": h.current_price,
                    "value": h.market_value,
                    "weight": (h.market_value / self.total_value) if self.total_value else 0.0,
                    "asset_class": h.asset_class,
                    "raw_exposure": h.raw_exposure,
                }
                for h in self.holdings
            ],
        }

    def save_analysis(self, output_path: str) -> None:
        with open(output_path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
        print(f"[+] Portfolio analysis saved to: {output_path}")