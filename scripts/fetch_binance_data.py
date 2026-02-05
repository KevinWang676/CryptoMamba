"""
Fetch Bitcoin OHLCV candlestick data from Binance API.

Usage:
    python scripts/fetch_binance_data.py --interval 15m --start 2023-06-01 --end 2026-02-05
    python scripts/fetch_binance_data.py --interval 30m --start 2023-06-01 --end 2026-02-05
    python scripts/fetch_binance_data.py --interval 1h  --start 2023-06-01 --end 2026-02-05

The script validates that adjacent rows have the exact expected time interval
and reports any gaps in the data.
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser


# Interval string to seconds mapping
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def fetch_binance_klines(symbol, interval, start_date, end_date):
    """Fetch historical klines from Binance API."""
    base_url = "https://api.binance.com/api/v3/klines"

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    all_data = []
    current_ts = start_ts

    while current_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "endTime": end_ts,
            "limit": 1000,
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text}")
            break

        data = response.json()
        if not data:
            break

        all_data.extend(data)
        current_ts = data[-1][6] + 1  # close_time + 1ms

        time.sleep(0.1)
        dt = datetime.fromtimestamp(current_ts / 1000).strftime("%Y-%m-%d %H:%M")
        print(f"\r  Fetched {len(all_data)} candles, up to {dt}", end="", flush=True)

    print()

    df = pd.DataFrame(all_data, columns=[
        "Timestamp", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_volume", "Trades",
        "Taker_buy_base", "Taker_buy_quote", "Ignore",
    ])

    df["Timestamp"] = df["Timestamp"] // 1000  # ms -> seconds
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df = df.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    return df


def validate_intervals(df, expected_seconds):
    """
    Validate that adjacent rows have exactly the expected time difference.
    Returns (is_valid, gaps) where gaps is a list of (timestamp, gap_seconds) tuples.
    """
    if len(df) < 2:
        return True, []

    timestamps = df["Timestamp"].values
    diffs = timestamps[1:] - timestamps[:-1]

    gaps = []
    for i, diff in enumerate(diffs):
        if diff != expected_seconds:
            gap_ts = int(timestamps[i])
            gaps.append((gap_ts, int(diff)))

    return len(gaps) == 0, gaps


def fill_gaps(df, expected_seconds):
    """
    Fill gaps in the data by forward-filling missing candles.
    This ensures every adjacent row has exactly expected_seconds difference.
    """
    if len(df) < 2:
        return df

    timestamps = df["Timestamp"].values
    start_ts = int(timestamps[0])
    end_ts = int(timestamps[-1])

    # Generate complete timestamp sequence
    complete_ts = list(range(start_ts, end_ts + 1, expected_seconds))

    # Create a complete dataframe
    complete_df = pd.DataFrame({"Timestamp": complete_ts})
    merged = complete_df.merge(df, on="Timestamp", how="left")

    # Forward fill missing values
    merged = merged.ffill()

    return merged


def main():
    parser = ArgumentParser(description="Fetch BTC OHLCV data from Binance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="15m",
                        choices=["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"])
    parser.add_argument("--start", type=str, default="2023-06-01")
    parser.add_argument("--end", type=str, default="2026-02-05")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path. Default: data/btc_{interval}.csv")
    parser.add_argument("--fill-gaps", action="store_true", default=False,
                        help="Fill missing candles by forward-filling previous values")
    parser.add_argument("--validate-only", action="store_true", default=False,
                        help="Only validate existing file, don't fetch new data")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/btc_{args.interval}.csv"

    expected_seconds = INTERVAL_SECONDS[args.interval]

    # Validate-only mode: check existing file
    if args.validate_only:
        if not os.path.exists(args.output):
            print(f"Error: File {args.output} does not exist")
            sys.exit(1)
        df = pd.read_csv(args.output)
        print(f"Validating {args.output} ({len(df)} rows)...")
        is_valid, gaps = validate_intervals(df, expected_seconds)
        if is_valid:
            print(f"  OK: All {len(df)-1} intervals are exactly {expected_seconds} seconds ({args.interval})")
        else:
            print(f"  WARNING: Found {len(gaps)} gaps:")
            for ts, diff in gaps[:10]:
                dt = datetime.fromtimestamp(ts)
                print(f"    {dt}: gap of {diff}s (expected {expected_seconds}s)")
            if len(gaps) > 10:
                print(f"    ... and {len(gaps) - 10} more gaps")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Fetching {args.symbol} {args.interval} data: {args.start} -> {args.end}")
    df = fetch_binance_klines(args.symbol, args.interval, args.start, args.end)

    # Validate intervals
    is_valid, gaps = validate_intervals(df, expected_seconds)
    if not is_valid:
        print(f"\nWARNING: Found {len(gaps)} gaps in data:")
        for ts, diff in gaps[:5]:
            dt = datetime.fromtimestamp(ts)
            expected_candles = diff // expected_seconds
            print(f"  {dt}: gap of {diff}s ({expected_candles} missing candles)")
        if len(gaps) > 5:
            print(f"  ... and {len(gaps) - 5} more gaps")

        if args.fill_gaps:
            print("\nFilling gaps with forward-fill...")
            original_len = len(df)
            df = fill_gaps(df, expected_seconds)
            print(f"  Added {len(df) - original_len} rows to fill gaps")
    else:
        print(f"Data validated: all {len(df)-1} intervals are exactly {expected_seconds}s")

    df.to_csv(args.output, index=False)
    start_dt = datetime.fromtimestamp(df["Timestamp"].iloc[0])
    end_dt = datetime.fromtimestamp(df["Timestamp"].iloc[-1])
    print(f"Saved {len(df)} candles to {args.output}")
    print(f"Range: {start_dt} -> {end_dt}")


if __name__ == "__main__":
    main()
