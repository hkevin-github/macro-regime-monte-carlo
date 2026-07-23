#!/usr/bin/env python3
"""
Debug script for inspecting how BIAPX is classified by the portfolio analyzer.

This script prints:
- Yahoo metadata
- raw fund payloads
- parsed holdings
- normalized exposures
- rolled-up broad asset classes

Use this to diagnose why a balanced fund is being classified incorrectly.
"""

from __future__ import annotations

import json
from pprint import pprint

import yfinance as yf

from portfolio_analyzer import TickerClassifier


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_json(obj):
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


def debug_ticker(ticker: str):
    classifier = TickerClassifier()
    ticker = ticker.upper()

    print_section(f"DEBUGGING TICKER: {ticker}")

    # 1) Fetch metadata
    metadata = classifier.fetch_metadata(ticker)
    print("[1] Metadata:")
    pprint(metadata)

    q = metadata.get("quoteType", "")
    cat = metadata.get("category", "")
    name = metadata.get("longName", "") + " " + metadata.get("shortName", "")
    text = f"{q} {cat} {metadata.get('fundFamily', '')} {metadata.get('sector', '')} {metadata.get('industry', '')} {name}".lower()

    print("\n[2] Derived text used for classification:")
    print(text)

    # 2) Print raw Yahoo info keys
    print_section("RAW YAHOO INFO")
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception as e:
        info = {}
        print(f"[!] Failed to fetch t.info: {e}")

    print(f"Info keys ({len(info)}):")
    print(sorted(info.keys()))

    interesting_keys = [
        "quoteType",
        "category",
        "fundFamily",
        "sector",
        "industry",
        "longName",
        "shortName",
        "currentPrice",
        "regularMarketPrice",
        "previousClose",
        "sectorWeightings",
        "topHoldings",
        "assetClass",
        "holdings",
    ]

    print("\nInteresting info fields:")
    for key in interesting_keys:
        if key in info:
            print(f"\n--- {key} ---")
            print(safe_json(info.get(key)))
        else:
            print(f"\n--- {key} ---")
            print("(missing)")

    # 3) Try fund holdings
    print_section("FUND LOOK-THROUGH")
    candidate_payloads = []

    for key in ["sectorWeightings", "topHoldings", "assetClass", "holdings", "fundHoldings"]:
        if key in info and info[key]:
            candidate_payloads.append((f"info.{key}", info[key]))

    for attr in ["funds_data"]:
        obj = getattr(t, attr, None)
        if obj is not None:
            for field in ["top_holdings", "sector_weightings", "asset_classes", "equity_holdings"]:
                try:
                    value = getattr(obj, field, None)
                    if value:
                        candidate_payloads.append((f"{attr}.{field}", value))
                except Exception as e:
                    print(f"[!] Error accessing {attr}.{field}: {e}")

    if not candidate_payloads:
        print("No candidate payloads found.")
    else:
        print(f"Found {len(candidate_payloads)} candidate payload(s):")
        for src, payload in candidate_payloads:
            print(f"\n--- SOURCE: {src} ---")
            print(f"Type: {type(payload)}")
            print(safe_json(payload))

    # 4) Parse each payload the same way the classifier does
    print_section("PARSED PAYLOADS")
    for src, payload in candidate_payloads:
        parsed = classifier._parse_holdings_payload(payload)
        print(f"\n--- {src} ---")
        print("Parsed:")
        pprint(parsed)

        if parsed:
            rolled = classifier._normalize_and_rollup_exposure(parsed)
            print("Rolled up to broad classes:")
            pprint(rolled)

    # 5) Metadata classification
    print_section("METADATA CLASSIFICATION")
    meta_classification = classifier.classify_metadata(metadata)
    print("Raw metadata classification:")
    pprint(meta_classification)

    rolled_meta = classifier._normalize_and_rollup_exposure(meta_classification)
    print("Rolled metadata classification:")
    pprint(rolled_meta)

    # 6) Full classifier result
    print_section("FINAL classify_ticker RESULT")
    result = classifier.classify_ticker(ticker)
    pprint(result)

    # 7) Balanced heuristic check
    print_section("BALANCED FUND HEURISTIC CHECK")
    balanced_hint = classifier._infer_balanced_exposure_from_text(text)
    print("Balanced hint:")
    pprint(balanced_hint)

    print_section("SUMMARY")
    print(f"Ticker: {ticker}")
    print(f"Category: {cat}")
    print(f"Quote type: {q}")
    print(f"Final classification: {result}")

    if "US_Equity" in result and result.get("US_Equity", 0.0) > 0.9:
        print("\n[WARNING] Fund is being classified as nearly all US equity.")
        print("This likely means Yahoo look-through is incomplete or the fallback is too aggressive.")

    if "US_Bonds" not in result and balanced_hint is not None:
        print("\n[WARNING] Balanced heuristic exists but bond sleeve did not appear in final classification.")
        print("You may need to prefer the balanced heuristic when look-through is ambiguous.")


if __name__ == "__main__":
    debug_ticker("VXUS")