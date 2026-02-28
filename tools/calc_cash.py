"""Calculate current cash balances from trade history.

Why: The cash-update feature was implemented after many trades were
     already recorded.  This script retroactively computes the correct
     JPY.CASH and USD.CASH values by applying all unsettled trades
     to the baseline balances.
How: 1) Parse brokerage CSV exports to identify trades whose settlement
        dates fall AFTER the baseline dates (JPY: 2026-02-19, USD: 2026-02-18).
     2) Parse JSON-only trades (entered after the CSV export).
     3) Sum the deltas and add to the baseline balances.
"""

from __future__ import annotations

import csv
import json
import pathlib

TRADE_DIR = pathlib.Path("data/history/trade")
JP_CSV = TRADE_DIR / "tradehistory(JP)_20260219.csv"
US_CSV = TRADE_DIR / "tradehistory(US)_20260219.csv"

# Baseline cash values set when portfolio was initialised
BASELINE_JPY = 10_694.0  # purchase_date = 2026-02-19
BASELINE_USD = 17_757.77  # purchase_date = 2026-02-18

# Batch import timestamp cutoff (initial CSV import)
BATCH_CUTOFF = "2026-02-19T23:12:00"


def _normalise_date(raw: str) -> str:
    """Normalise 'YYYY/M/D' or 'YYYY-M-D' to 'YYYY-MM-DD' for safe comparison."""
    parts = raw.strip('"').replace("/", "-").split("-")
    return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"


def _parse_jp_csv_unsettled() -> list[dict]:
    """Return JP CSV trades that settled AFTER the JPY baseline date."""
    results: list[dict] = []
    with open(JP_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            trade_date = _normalise_date(row[0])
            settle_date = _normalise_date(row[1])
            code = row[2].strip('"')
            side = row[7].strip('"')
            settle_str = row[16].strip('"').replace(",", "")
            settle_amt = float(settle_str) if settle_str not in ("-", "") else 0.0

            if settle_date > "2026-02-19":
                results.append(
                    {
                        "trade_date": trade_date,
                        "settle_date": settle_date,
                        "code": code,
                        "side": side,
                        "settle_jpy": settle_amt,
                    }
                )
    return results


def _parse_us_csv_unsettled() -> list[dict]:
    """Return US CSV trades that settled AFTER their respective baselines."""
    results: list[dict] = []
    with open(US_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            trade_date = _normalise_date(row[0])
            settle_date = _normalise_date(row[1])
            ticker = row[2].strip('"')
            trade_type_raw = row[5].strip('"')
            side = row[6].strip('"')
            settle_cur = row[9].strip('"')
            settle_usd_str = row[16].strip('"').replace(",", "")
            settle_jpy_str = row[17].strip('"').replace(",", "")
            settle_usd = (
                float(settle_usd_str) if settle_usd_str not in ("-", "") else 0.0
            )
            settle_jpy = (
                float(settle_jpy_str) if settle_jpy_str not in ("-", "") else 0.0
            )

            is_usd_settled = "ドル" in settle_cur

            # Skip non-trade entries (入庫 = transfer-in)
            if side in ("", "入庫") or trade_type_raw == "入庫":
                continue

            # USD-settled trades: check against USD baseline (2026-02-18)
            if is_usd_settled and settle_date > "2026-02-18" and settle_usd > 0:
                results.append(
                    {
                        "trade_date": trade_date,
                        "settle_date": settle_date,
                        "ticker": ticker,
                        "side": side,
                        "currency": "USD",
                        "amount": settle_usd,
                    }
                )
            # JPY-settled trades: check against JPY baseline (2026-02-19)
            elif (
                not is_usd_settled
                and settle_jpy > 0
                and settle_date > "2026-02-19"
            ):
                results.append(
                    {
                        "trade_date": trade_date,
                        "settle_date": settle_date,
                        "ticker": ticker,
                        "side": side,
                        "currency": "JPY",
                        "amount": settle_jpy,
                    }
                )
    return results


def _process_json_only_trades() -> tuple[float, float, float]:
    """Process JSON trades not in the CSV exports.

    Returns (json_jpy_delta, json_usd_delta, batch_fallback_jpy_delta).
    """
    json_jpy = 0.0
    json_usd = 0.0
    batch_fallback_jpy = 0.0

    json_files = sorted(TRADE_DIR.glob("*.json"))
    for f_path in json_files:
        d = json.loads(f_path.read_text(encoding="utf-8"))
        saved = d.get("_saved_at", "")
        memo = d.get("memo", "")
        trade_type = d.get("trade_type", "")
        symbol = d.get("symbol", "")

        if trade_type == "transfer" or symbol.endswith(".CASH"):
            continue

        if saved <= BATCH_CUTOFF:
            # Batch import: only handle trades with missing settlement data
            # that are in the CSV but have settle_amount = "-"
            if (
                d["date"] >= "2026-02-19"
                and d.get("settlement_jpy", 0) == 0
                and d.get("settlement_usd", 0) == 0
            ):
                fallback = d["shares"] * d["price"]
                delta = fallback if trade_type == "sell" else -fallback
                batch_fallback_jpy += delta
                print(
                    f"  [BATCH-FALLBACK] {d['date']} {trade_type} {symbol}"
                    f" JPY {delta:+,.0f} ({d['shares']}x{d['price']})"
                )
            continue

        # Non-batch trades
        s_jpy = d.get("settlement_jpy", 0.0)
        s_usd = d.get("settlement_usd", 0.0)
        shares = d.get("shares", 0)
        price = d.get("price", 0.0)
        currency = d["currency"]

        if s_jpy > 0:
            delta = s_jpy if trade_type == "sell" else -s_jpy
            json_jpy += delta
            print(f"  {d['date']} {trade_type} {symbol} JPY {delta:+,.0f}")
        if s_usd > 0:
            delta = s_usd if trade_type == "sell" else -s_usd
            json_usd += delta
            print(f"  {d['date']} {trade_type} {symbol} USD {delta:+,.2f}")
        if s_jpy == 0 and s_usd == 0:
            fallback = shares * price
            if fallback > 0:
                delta = fallback if trade_type == "sell" else -fallback
                if currency == "JPY":
                    json_jpy += delta
                elif currency == "USD":
                    json_usd += delta
                print(
                    f"  {d['date']} {trade_type} {symbol}"
                    f" {currency} {delta:+,.0f} (fallback)"
                )

    return json_jpy, json_usd, batch_fallback_jpy


def main() -> None:
    """Calculate and display updated cash balances."""
    # Step 1: CSV-based unsettled trades
    jp_unsettled = _parse_jp_csv_unsettled()
    us_unsettled = _parse_us_csv_unsettled()

    print("=== JP CSV: trades settling AFTER baseline (2026-02-19) ===")
    for t in jp_unsettled:
        sign = "+" if t["side"] == "売付" else "-"
        amt = f"{t['settle_jpy']:,.0f}" if t["settle_jpy"] > 0 else "(missing)"
        print(
            f"  {t['trade_date']} {t['side']} {t['code']}"
            f" settle={t['settle_date']} JPY {sign}{amt}"
        )

    print()
    print("=== US CSV: trades settling AFTER baseline ===")
    for t in us_unsettled:
        sign = "+" if t["side"] == "売付" else "-"
        print(
            f"  {t['trade_date']} {t['side']} {t['ticker']}"
            f" settle={t['settle_date']} {t['currency']} {sign}{t['amount']:,.2f}"
        )

    # Step 2: Calculate CSV delta
    csv_jpy = 0.0
    csv_usd = 0.0
    for t in jp_unsettled:
        if t["settle_jpy"] > 0:
            csv_jpy += t["settle_jpy"] if t["side"] == "売付" else -t["settle_jpy"]
    for t in us_unsettled:
        if t["currency"] == "USD":
            csv_usd += t["amount"] if t["side"] == "売付" else -t["amount"]
        else:
            csv_jpy += t["amount"] if t["side"] == "売付" else -t["amount"]

    print(f"\nCSV unsettled delta: JPY={csv_jpy:+,.0f}  USD={csv_usd:+,.2f}")

    # Step 3: JSON-only trades
    print("\n=== JSON-only trades (post-CSV) ===")
    json_jpy, json_usd, batch_fallback = _process_json_only_trades()
    csv_jpy += batch_fallback  # Add fallback for batch trades with missing amounts

    print(f"\nJSON-only delta: JPY={json_jpy:+,.0f}  USD={json_usd:+,.2f}")
    if batch_fallback != 0:
        print(f"Batch fallback:  JPY={batch_fallback:+,.0f}")

    # Step 4: Final calculation
    total_jpy = csv_jpy + json_jpy
    total_usd = csv_usd + json_usd
    new_jpy = BASELINE_JPY + total_jpy
    new_usd = BASELINE_USD + total_usd

    print()
    print("=" * 60)
    print(f"Baseline:      JPY.CASH = {BASELINE_JPY:>12,.2f}")
    print(f"               USD.CASH = {BASELINE_USD:>12,.2f}")
    print(f"CSV delta:     JPY = {csv_jpy:>+13,.0f}")
    print(f"               USD = {csv_usd:>+13,.2f}")
    print(f"JSON delta:    JPY = {json_jpy:>+13,.0f}")
    print(f"               USD = {json_usd:>+13,.2f}")
    print(f"Total delta:   JPY = {total_jpy:>+13,.0f}")
    print(f"               USD = {total_usd:>+13,.2f}")
    print()
    print(f">>> NEW JPY.CASH = {new_jpy:>12,.2f}")
    print(f">>> NEW USD.CASH = {new_usd:>12,.2f}")


if __name__ == "__main__":
    main()
