"""
Baseline models (XGBoost, LightGBM, Logistic Regression, Ensemble) for
BTC next-candle up/down classification.

Uses the same data and train/val/test splits as CryptoMamba for fair comparison.

Usage:
    python scripts/baseline_models.py --interval 15m
    python scripts/baseline_models.py --interval 30m
    python scripts/baseline_models.py --interval 1h
"""

import os
import sys
import pathlib

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    print("Please install xgboost: pip install xgboost")
    sys.exit(1)

try:
    import lightgbm as lgb
except ImportError:
    print("Please install lightgbm: pip install lightgbm")
    sys.exit(1)

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

ROOT = os.path.dirname(pathlib.Path(__file__).parent.absolute())

# ---------------------------------------------------------------------------
# Config — matches configs/data_configs/*.yaml exactly
# ---------------------------------------------------------------------------
INTERVAL_CONFIG = {
    "15m": {
        "data_path": f"{ROOT}/data/btc_15m.csv",
        "jumps": 900,
        "train_interval": ["2023-06-01 00:00:00", "2025-06-01 00:00:00"],
        "val_interval": ["2025-06-01 00:00:00", "2025-10-01 00:00:00"],
        "test_interval": ["2025-10-01 00:00:00", "2026-02-05 00:00:00"],
    },
    "30m": {
        "data_path": f"{ROOT}/data/btc_30m.csv",
        "jumps": 1800,
        "train_interval": ["2023-06-01 00:00:00", "2025-06-01 00:00:00"],
        "val_interval": ["2025-06-01 00:00:00", "2025-10-01 00:00:00"],
        "test_interval": ["2025-10-01 00:00:00", "2026-02-05 00:00:00"],
    },
    "1h": {
        "data_path": f"{ROOT}/data/btc_1h.csv",
        "jumps": 3600,
        "train_interval": ["2023-06-01 00:00:00", "2025-06-01 00:00:00"],
        "val_interval": ["2025-06-01 00:00:00", "2025-10-01 00:00:00"],
        "test_interval": ["2025-10-01 00:00:00", "2026-02-05 00:00:00"],
    },
}


def get_args():
    parser = ArgumentParser(description="Baseline models for BTC up/down classification")
    parser.add_argument("--interval", type=str, default="15m",
                        choices=["15m", "30m", "1h"])
    parser.add_argument("--save_models", action="store_true", default=False,
                        help="Save trained models to checkpoints/baselines/")
    parser.add_argument("--use_weights", action="store_true", default=False,
                        help="Enable sample weighting by return magnitude")
    parser.add_argument("--tune", action="store_true", default=False,
                        help="Use Optuna to tune LightGBM/CatBoost/XGBoost hyperparameters")
    parser.add_argument("--tune_trials", type=int, default=100,
                        help="Number of Optuna trials (default: 100)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Multi-Timeframe Feature Engineering
# ---------------------------------------------------------------------------
# Use completed higher-timeframe candles as context for base-timeframe
# predictions.  Each base row gets features from the LAST COMPLETED HTF
# candle (shifted by 1 period) to prevent any data leakage.

HTF_CONFIGS = {
    900:  [("30m", 1800), ("1h", 3600), ("4h", 14400)],   # 15m base
    1800: [("1h", 3600), ("4h", 14400)],                    # 30m base
    3600: [("4h", 14400), ("1d", 86400)],                   # 1h base
}


def resample_to_htf(df, period_seconds):
    """Resample OHLCV data to a higher timeframe by grouping timestamps."""
    htf_ts = (df["Timestamp"] // period_seconds) * period_seconds
    htf = df.groupby(htf_ts).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    return htf


def compute_htf_indicators(htf, prefix):
    """Compute ~19 technical indicators on higher-timeframe OHLCV data.

    Features cover trend, momentum, volatility, volume, and position.
    All use only past HTF data (rolling lookback windows).
    """
    f = pd.DataFrame(index=htf.index)
    c = htf["Close"]
    h = htf["High"]
    l = htf["Low"]
    o = htf["Open"]
    v = htf["Volume"]

    # Trend: SMA & EMA ratios
    for p in [5, 10, 20]:
        f[f"{prefix}_sma{p}"] = c / c.rolling(p).mean() - 1
    for p in [10, 20]:
        f[f"{prefix}_ema{p}"] = c / c.ewm(span=p, adjust=False).mean() - 1

    # Momentum: RSI
    for p in [7, 14]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss_v = (-delta.clip(upper=0)).rolling(p).mean()
        f[f"{prefix}_rsi{p}"] = 100 - 100 / (1 + gain / loss_v.replace(0, 1e-10))

    # MACD (normalized by price)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    f[f"{prefix}_macd"] = macd / c
    f[f"{prefix}_macd_h"] = (macd - sig) / c

    # Volatility: ATR, Bollinger Bands, H-L range
    prev_c = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    f[f"{prefix}_atr14"] = tr.rolling(14).mean() / c
    bb_m = c.rolling(20).mean()
    bb_s = c.rolling(20).std()
    bb_w = 4 * bb_s
    f[f"{prefix}_bb_pos"] = (c - (bb_m - 2 * bb_s)) / bb_w.replace(0, 1e-10)
    f[f"{prefix}_bb_w"] = bb_w / bb_m
    f[f"{prefix}_hlr"] = (h - l) / c

    # Returns
    f[f"{prefix}_ret1"] = c.pct_change(1)
    f[f"{prefix}_ret4"] = c.pct_change(4)

    # Volume ratio
    f[f"{prefix}_vr"] = v / v.rolling(10).mean().replace(0, 1e-10)

    # Position in recent range
    for p in [10, 20]:
        hh = h.rolling(p).max()
        ll = l.rolling(p).min()
        f[f"{prefix}_rng{p}"] = (c - ll) / (hh - ll).replace(0, 1e-10)

    # Candle direction
    f[f"{prefix}_bull"] = (c > o).astype(int)

    return f.replace([np.inf, -np.inf], np.nan)


def add_htf_features(df, base_interval_seconds=900):
    """Add higher-timeframe features to base OHLCV data.

    For each configured HTF:
      1. Resamples base data to HTF candles
      2. Computes technical indicators on HTF
      3. Shifts by 1 HTF period (only uses COMPLETED candles — no leakage)
      4. Maps features back to base-timeframe rows via timestamp grouping

    Returns DataFrame with HTF features, indexed same as df.
    """
    configs = HTF_CONFIGS.get(base_interval_seconds, [])
    if not configs:
        return pd.DataFrame(index=df.index)

    names = [prefix for prefix, _ in configs]
    print(f"  Adding multi-timeframe features: {', '.join(names)}")

    parts = []
    for prefix, period_sec in configs:
        htf = resample_to_htf(df, period_sec)
        indicators = compute_htf_indicators(htf, prefix)
        # Shift by 1: each row now holds the PREVIOUS completed candle's features
        shifted = indicators.shift(1)
        # Map back to base rows: each base row → its HTF group → shifted features
        base_htf_ts = (df["Timestamp"].values // period_sec) * period_sec
        mapped = shifted.reindex(base_htf_ts)
        mapped.index = df.index
        parts.append(mapped)
        print(f"    {prefix}: {len(indicators.columns)} features "
              f"({len(htf)} candles)")

    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df, base_interval_seconds=900):
    """Create tabular features from OHLCV data for tree-based models.

    All features use only past data (no future leakage).
    The target is computed from the NEXT candle's Close vs current Close.
    Higher-timeframe features are added for multi-scale context.
    """
    features = pd.DataFrame(index=df.index)

    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    returns = close.pct_change()

    # --- 1. Price Returns at various lags ---
    for lag in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96]:
        features[f"ret_{lag}"] = close.pct_change(lag)

    # --- 2. SMA ratios ---
    for period in [5, 10, 20, 50]:
        sma = close.rolling(period).mean()
        features[f"sma_ratio_{period}"] = close / sma - 1

    # SMA crossover
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    features["sma_cross_5_20"] = sma5 / sma20 - 1

    # --- 3. EMA ratios ---
    for period in [5, 10, 20, 50]:
        ema = close.ewm(span=period, adjust=False).mean()
        features[f"ema_ratio_{period}"] = close / ema - 1

    # --- 4. RSI ---
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss_val = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss_val.replace(0, 1e-10)
        features[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # --- 5. MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    features["macd"] = macd / close
    features["macd_signal"] = signal / close
    features["macd_hist"] = (macd - signal) / close

    # --- 6. Bollinger Bands ---
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    bb_width = bb_upper - bb_lower
    features["bb_position"] = (close - bb_lower) / bb_width.replace(0, 1e-10)
    features["bb_width"] = bb_width / bb_sma

    # --- 7. Volatility ---
    features["hl_range"] = (high - low) / close
    features["oc_range"] = (close - open_) / close
    for period in [5, 10, 20]:
        features[f"volatility_{period}"] = returns.rolling(period).std()

    # --- 8. ATR (Average True Range) ---
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    for period in [7, 14]:
        features[f"atr_{period}"] = tr.rolling(period).mean() / close

    # --- 9. Volume features ---
    for period in [5, 10, 20]:
        vol_sma = volume.rolling(period).mean()
        features[f"vol_ratio_{period}"] = volume / vol_sma.replace(0, 1e-10)
    features["vol_change_1"] = volume.pct_change(1)
    features["vol_change_4"] = volume.pct_change(4)

    # --- 10. Candlestick features ---
    body = (close - open_).abs()
    total_range = (high - low).replace(0, 1e-10)
    candle_max = np.maximum(close, open_)
    candle_min = np.minimum(close, open_)
    features["body_ratio"] = body / total_range
    features["upper_shadow"] = (high - candle_max) / total_range
    features["lower_shadow"] = (candle_min - low) / total_range
    features["bullish_candle"] = (close > open_).astype(int)

    # --- 11. Momentum ---
    for period in [4, 8, 16, 32]:
        features[f"momentum_{period}"] = close / close.shift(period) - 1

    # --- 12. Lagged candle features ---
    for lag in [1, 2, 3, 4]:
        features[f"lag_{lag}_ret"] = close / close.shift(lag) - 1
        features[f"lag_{lag}_vol_ratio"] = volume / volume.shift(lag).replace(0, 1e-10)
        features[f"lag_{lag}_hl_range"] = (high.shift(lag) - low.shift(lag)) / close.shift(lag)

    # --- 13. Sequential pattern features (tree models don't see order) ---
    # Consecutive up/down candle count
    bullish = (close > close.shift(1)).astype(int)
    bearish = (close < close.shift(1)).astype(int)
    # Count consecutive ups: reset counter on each down candle
    consec_up = bullish.copy()
    consec_down = bearish.copy()
    for i in range(1, len(consec_up)):
        if bullish.iloc[i] == 1:
            consec_up.iloc[i] = consec_up.iloc[i - 1] + 1
        if bearish.iloc[i] == 1:
            consec_down.iloc[i] = consec_down.iloc[i - 1] + 1
    features["consec_up"] = consec_up
    features["consec_down"] = consec_down

    # Williams %R (position relative to recent high/low)
    for period in [14, 28]:
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        features[f"williams_r_{period}"] = (hh - close) / (hh - ll).replace(0, 1e-10)

    # Price relative to recent high/low
    for period in [24, 48, 96]:
        features[f"dist_from_high_{period}"] = close / high.rolling(period).max() - 1
        features[f"dist_from_low_{period}"] = close / low.rolling(period).min() - 1

    # Rolling mean of returns (trend direction signal)
    for period in [5, 10, 20]:
        features[f"ret_mean_{period}"] = returns.rolling(period).mean()

    # --- 14. Time-of-day (cyclical encoding) ---
    if "Timestamp" in df.columns:
        dt = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
        hour = dt.dt.hour + dt.dt.minute / 60
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        day_of_week = dt.dt.dayofweek
        features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    # --- 15. Buy/Sell Volume Pressure ---
    up_vol = volume.where(close > close.shift(1), 0)
    down_vol = volume.where(close < close.shift(1), 0)
    for p in [10, 20]:
        up_sum = up_vol.rolling(p).sum()
        down_sum = down_vol.rolling(p).sum()
        total = (up_sum + down_sum).replace(0, 1e-10)
        features[f"buy_pressure_{p}"] = up_sum / total - 0.5  # centered at 0

    # --- 16. VWAP Deviation ---
    typical_price = (high + low + close) / 3
    for p in [10, 20]:
        vwap = (typical_price * volume).rolling(p).sum() / volume.rolling(p).sum().replace(0, 1e-10)
        features[f"vwap_dev_{p}"] = (close - vwap) / close

    # --- 17. Gap (open vs previous close) ---
    features["gap"] = (open_ - close.shift(1)) / close.shift(1)

    # --- 18. Price Efficiency ---
    features["price_efficiency"] = (close - open_).abs() / (high - low).replace(0, 1e-10)

    # --- 19. On-Balance Volume (OBV) ---
    obv_sign = np.sign(close.diff()).fillna(0)
    obv = (obv_sign * volume).cumsum()
    # Normalize OBV: ratio to its own EMA (captures divergence)
    obv_ema20 = obv.ewm(span=20, adjust=False).mean()
    features["obv_ema_ratio"] = obv / obv_ema20.replace(0, 1e-10) - 1
    # OBV slope (change over recent window, normalized by volume)
    obv_mean_vol = volume.rolling(20).mean().replace(0, 1e-10)
    features["obv_slope_10"] = obv.diff(10) / (10 * obv_mean_vol)

    # --- 20. Money Flow Index (MFI) — volume-weighted RSI ---
    typical_price_mfi = (high + low + close) / 3
    raw_money_flow = typical_price_mfi * volume
    mfi_delta = typical_price_mfi.diff()
    pos_flow = raw_money_flow.where(mfi_delta > 0, 0)
    neg_flow = raw_money_flow.where(mfi_delta < 0, 0)
    pos_sum = pos_flow.rolling(14).sum()
    neg_sum = neg_flow.rolling(14).sum().replace(0, 1e-10)
    money_ratio = pos_sum / neg_sum
    features["mfi_14"] = 100 - 100 / (1 + money_ratio)

    # --- Multi-Timeframe Features ---
    htf_feats = add_htf_features(df, base_interval_seconds)
    features = pd.concat([features, htf_feats], axis=1)

    # --- Target: next candle goes UP ---
    next_close = close.shift(-1)
    features["target"] = np.where(next_close.notna(), (next_close > close).astype(int), np.nan)

    # --- Return magnitude (metadata for sample weighting, NOT a training feature) ---
    features["_return_mag"] = np.abs(next_close / close - 1)

    # --- Clean up inf values ---
    features = features.replace([np.inf, -np.inf], np.nan)

    return features


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def clip_probs(y_prob, eps=1e-6):
    """Clip probabilities to avoid numerical issues in log-loss."""
    return np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)


def evaluate(y_true, y_pred, y_prob, name):
    """Print evaluation metrics and return accuracy."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    up_ratio = y_true.mean()

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  UP ratio:    {up_ratio:.4f} ({int(y_true.sum())}/{len(y_true)})")
    print(f"  Pred UP:     {y_pred.mean():.4f} ({int(y_pred.sum())}/{len(y_pred)})")

    if y_prob is not None:
        try:
            y_prob = clip_probs(y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ll = log_loss(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            print(f"  AUC-ROC:     {auc:.4f}")
            print(f"  Log Loss:    {ll:.4f}")
            print(f"  Brier Score: {brier:.4f}")
        except Exception:
            pass

    print(f"{'=' * 60}")
    return acc


def optimize_ensemble_weights(prob_matrix, y_true, n_iter=4000, seed=42):
    """Search convex blend weights that minimize validation log-loss."""
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)
    n_models = prob_matrix.shape[1]

    # Deterministic candidates first
    candidates = [np.ones(n_models, dtype=np.float64) / n_models]
    for i in range(n_models):
        w = np.zeros(n_models, dtype=np.float64)
        w[i] = 1.0
        candidates.append(w)

    # Random simplex search for better mixtures
    rng = np.random.default_rng(seed)
    random_weights = rng.dirichlet(alpha=np.ones(n_models), size=n_iter)

    best_weights = candidates[0]
    best_logloss = float("inf")

    for w in candidates:
        pred = clip_probs(prob_matrix @ w)
        ll = log_loss(y_true, pred)
        if ll < best_logloss:
            best_logloss = ll
            best_weights = w

    for w in random_weights:
        pred = clip_probs(prob_matrix @ w)
        ll = log_loss(y_true, pred)
        if ll < best_logloss:
            best_logloss = ll
            best_weights = w

    return best_weights, best_logloss


# ---------------------------------------------------------------------------
# Threshold-based betting analysis
# ---------------------------------------------------------------------------
def threshold_analysis(y_true, y_prob, name, num_days, thresholds=None):
    """Analyze accuracy at various confidence thresholds.

    A bet is placed when max(prob_up, prob_down) > threshold,
    i.e., |prob_up - 0.5| > (threshold - 0.5).

    Args:
        y_true: actual labels (0/1)
        y_prob: predicted probability of UP
        name: model name for display
        num_days: number of calendar days in this split (for bets/day)
        thresholds: list of thresholds to test

    Returns:
        List of (threshold, accuracy, num_bets, bets_per_day) tuples
    """
    if thresholds is None:
        thresholds = [0.50, 0.51, 0.52, 0.53, 0.54, 0.545, 0.55,
                      0.555, 0.56, 0.57, 0.58, 0.60, 0.62, 0.65, 0.70]

    results = []
    for t in thresholds:
        # Bet when confidence > threshold (either direction)
        confidence = np.abs(y_prob - 0.5) + 0.5  # max(prob_up, prob_down)
        bet_mask = confidence > t
        n_bets = bet_mask.sum()

        if n_bets == 0:
            results.append((t, np.nan, 0, 0.0))
            continue

        y_bet = y_true[bet_mask]
        pred_bet = (y_prob[bet_mask] > 0.5).astype(int)
        acc = accuracy_score(y_bet, pred_bet)
        bets_per_day = n_bets / max(num_days, 1)

        results.append((t, acc, int(n_bets), bets_per_day))

    return results


def print_threshold_table(all_results, num_days_val, num_days_test):
    """Print threshold analysis table for all models."""
    model_names = list(all_results.keys())

    print(f"\n{'=' * 90}")
    print(f"  THRESHOLD-BASED BETTING ANALYSIS")
    print(f"  Only bet when model confidence > threshold")
    print(f"{'=' * 90}")

    for model_name in model_names:
        val_results, test_results = all_results[model_name]

        print(f"\n  --- {model_name} ---")
        header = (f"  {'Threshold':>9s} | {'Val Acc':>8s} {'Val Bets':>9s} "
                  f"{'Val/Day':>8s} | {'Test Acc':>8s} {'Test Bets':>10s} "
                  f"{'Test/Day':>9s}")
        print(header)
        print(f"  {'-' * 9} | {'-' * 8} {'-' * 9} {'-' * 8} | "
              f"{'-' * 8} {'-' * 10} {'-' * 9}")

        for (vt, vacc, vn, vpd), (tt, tacc, tn, tpd) in zip(val_results, test_results):
            vacc_s = f"{vacc:.4f}" if not np.isnan(vacc) else "   N/A"
            tacc_s = f"{tacc:.4f}" if not np.isnan(tacc) else "   N/A"
            print(f"  {vt:>9.2f} | {vacc_s:>8s} {vn:>9d} {vpd:>8.1f} | "
                  f"{tacc_s:>8s} {tn:>10d} {tpd:>9.1f}")

    # Find optimal threshold per model at different bet frequency targets
    # Use val set to select, report test set performance
    min_bets_configs = [
        ("Conservative (5-15/day)", 5.0, 15.0),
        ("Moderate (10-30/day)", 10.0, 30.0),
        ("Aggressive (20-50/day)", 20.0, 50.0),
    ]

    print(f"\n  {'=' * 78}")
    print(f"  RECOMMENDED THRESHOLDS (selected on val, reported on test)")
    print(f"  {'=' * 78}")

    for style_name, min_bpd, max_bpd in min_bets_configs:
        print(f"\n  {style_name}:")
        print(f"  {'Model':<20s} {'Threshold':>9s} {'Val Acc':>9s} "
              f"{'Test Acc':>9s} {'Test Bets':>10s} {'Test/Day':>9s} {'Edge':>7s}")
        print(f"  {'-' * 20} {'-' * 9} {'-' * 9} {'-' * 9} "
              f"{'-' * 10} {'-' * 9} {'-' * 7}")

        for model_name in model_names:
            val_results, test_results = all_results[model_name]
            best_vacc = -1
            best_idx = -1
            for i, (t, vacc, vn, vpd) in enumerate(val_results):
                if np.isnan(vacc) or vpd < min_bpd or vpd > max_bpd:
                    continue
                if vacc > best_vacc:
                    best_vacc = vacc
                    best_idx = i

            if best_idx < 0:
                print(f"  {model_name:<20s}  (no threshold in range)")
                continue

            vt, vacc, vn, vpd = val_results[best_idx]
            tt, tacc, tn, tpd = test_results[best_idx]
            vacc_s = f"{vacc:.4f}" if not np.isnan(vacc) else "N/A"
            tacc_s = f"{tacc:.4f}" if not np.isnan(tacc) else "N/A"
            edge = f"{(tacc - 0.5) * 100:+.1f}%" if not np.isnan(tacc) else "N/A"
            print(f"  {model_name:<20s} {vt:>9.3f} {vacc_s:>9s} "
                  f"{tacc_s:>9s} {tn:>10d} {tpd:>9.1f} {edge:>7s}")

    print(f"\n  {'=' * 78}")


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Tuning
# ---------------------------------------------------------------------------
def tune_lgb_optuna(X_train, y_train, X_val, y_val, n_trials=100,
                    sample_weight=None):
    """Tune LightGBM hyperparameters with Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 3000,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        val_prob = model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob > 0.5).astype(int)
        return accuracy_score(y_val, val_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n  Optuna LightGBM — Best val accuracy: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def tune_cat_optuna(X_train, y_train, X_val, y_val, n_trials=100,
                    sample_weight=None):
    """Tune CatBoost hyperparameters with Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "iterations": 3000,
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_seed": 42,
            "verbose": 0,
            "early_stopping_rounds": 50,
            "eval_metric": "Logloss",
        }
        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=(X_val, y_val),
        )
        val_prob = model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob > 0.5).astype(int)
        return accuracy_score(y_val, val_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n  Optuna CatBoost — Best val accuracy: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def tune_xgb_optuna(X_train, y_train, X_val, y_val, n_trials=100,
                    sample_weight=None):
    """Tune XGBoost hyperparameters with Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 3000,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "eval_metric": "logloss",
            "early_stopping_rounds": 50,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )
        val_prob = model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob > 0.5).astype(int)
        return accuracy_score(y_val, val_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n  Optuna XGBoost — Best val accuracy: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    config = INTERVAL_CONFIG[args.interval]
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading data from {config['data_path']}...")
    df = pd.read_csv(config["data_path"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print(f"  Total rows: {len(df)}")

    # ------------------------------------------------------------------
    # 2. Engineer features
    # ------------------------------------------------------------------
    print("Engineering features...")
    features = engineer_features(df, base_interval_seconds=config["jumps"])
    features = features.dropna()
    feature_cols = [c for c in features.columns if c != "target" and not c.startswith("_")]
    print(f"  Features: {len(feature_cols)} columns, {len(features)} samples")

    # ------------------------------------------------------------------
    # 3. Split by time intervals (same as CryptoMamba configs)
    # ------------------------------------------------------------------
    ts = df["Timestamp"].values[features.index]

    def to_ts(s):
        return int(datetime.strptime(s, date_fmt).timestamp())

    train_s, train_e = to_ts(config["train_interval"][0]), to_ts(config["train_interval"][1])
    val_s, val_e = to_ts(config["val_interval"][0]), to_ts(config["val_interval"][1])
    test_s, test_e = to_ts(config["test_interval"][0]), to_ts(config["test_interval"][1])

    train_mask = (ts >= train_s) & (ts < train_e)
    val_mask = (ts >= val_s) & (ts < val_e)
    test_mask = (ts >= test_s) & (ts < test_e)

    X_train = features.loc[features.index[train_mask], feature_cols].values
    y_train = features.loc[features.index[train_mask], "target"].values
    X_val = features.loc[features.index[val_mask], feature_cols].values
    y_val = features.loc[features.index[val_mask], "target"].values
    X_test = features.loc[features.index[test_mask], feature_cols].values
    y_test = features.loc[features.index[test_mask], "target"].values

    print(f"\n  Train: {len(X_train):>6d} samples  (UP ratio: {y_train.mean():.3f})")
    print(f"  Val:   {len(X_val):>6d} samples  (UP ratio: {y_val.mean():.3f})")
    print(f"  Test:  {len(X_test):>6d} samples  (UP ratio: {y_test.mean():.3f})")

    # ------------------------------------------------------------------
    # 3b. Compute sample weights (optional, off by default)
    # ------------------------------------------------------------------
    if args.use_weights:
        return_mag_train = features.loc[features.index[train_mask], "_return_mag"].values
        train_weights = np.clip(return_mag_train / 0.001, 0.2, 5.0)
        train_weights = train_weights / train_weights.mean()  # normalize to mean=1
        print(f"\n  Sample weights: min={train_weights.min():.2f}, "
              f"max={train_weights.max():.2f}, mean={train_weights.mean():.2f}")
    else:
        train_weights = None
        print("\n  Sample weights: DISABLED (use --use_weights to enable)")

    results = {}  # name -> (test_acc, test_prob, val_prob)

    # ==================================================================
    # 4. Logistic Regression (simple baseline)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Training Logistic Regression...")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=42,
    )
    lr_model.fit(X_train_scaled, y_train)

    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_acc = evaluate(y_test, lr_pred, lr_prob, f"Logistic Regression - {args.interval} Test")

    lr_val_pred = lr_model.predict(X_val_scaled)
    lr_val_prob = lr_model.predict_proba(X_val_scaled)[:, 1]
    evaluate(y_val, lr_val_pred, lr_val_prob, f"Logistic Regression - {args.interval} Val")
    results["LogisticRegression"] = (lr_acc, lr_prob, lr_val_prob)

    # ==================================================================
    # 5. XGBoost (with optional Optuna tuning)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Training XGBoost...")
    print("=" * 60)

    if args.tune:
        print("  Running Optuna hyperparameter search...")
        xgb_best_params = tune_xgb_optuna(
            X_train, y_train, X_val, y_val,
            n_trials=args.tune_trials, sample_weight=train_weights,
        )
        xgb_model = xgb.XGBClassifier(
            n_estimators=3000,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=50,
            **xgb_best_params,
        )
    else:
        xgb_model = xgb.XGBClassifier(
            n_estimators=1200,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=10,
            gamma=0.2,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=50,
        )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=train_weights,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_acc = evaluate(y_test, xgb_pred, xgb_prob, f"XGBoost - {args.interval} Test")

    xgb_val_pred = xgb_model.predict(X_val)
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    evaluate(y_val, xgb_val_pred, xgb_val_prob, f"XGBoost - {args.interval} Val")
    results["XGBoost"] = (xgb_acc, xgb_prob, xgb_val_prob)

    # ==================================================================
    # 6. LightGBM (with optional Optuna tuning)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Training LightGBM...")
    print("=" * 60)

    if args.tune:
        print("  Running Optuna hyperparameter search...")
        lgb_best_params = tune_lgb_optuna(
            X_train, y_train, X_val, y_val,
            n_trials=args.tune_trials, sample_weight=train_weights,
        )
        lgb_model = lgb.LGBMClassifier(
            n_estimators=3000,
            random_state=42,
            verbose=-1,
            **lgb_best_params,
        )
    else:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=2000,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
    lgb_model.fit(
        X_train, y_train,
        sample_weight=train_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ],
    )

    lgb_pred = lgb_model.predict(X_test)
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    lgb_acc = evaluate(y_test, lgb_pred, lgb_prob, f"LightGBM - {args.interval} Test")

    lgb_val_pred = lgb_model.predict(X_val)
    lgb_val_prob = lgb_model.predict_proba(X_val)[:, 1]
    evaluate(y_val, lgb_val_pred, lgb_val_prob, f"LightGBM - {args.interval} Val")
    results["LightGBM"] = (lgb_acc, lgb_prob, lgb_val_prob)

    # ==================================================================
    # 6.5. CatBoost
    # ==================================================================
    cat_prob = None
    cat_val_prob = None

    if HAS_CATBOOST:
        print("\n" + "=" * 60)
        print("  Training CatBoost...")
        print("=" * 60)

        if args.tune:
            print("  Running Optuna hyperparameter search...")
            cat_best_params = tune_cat_optuna(
                X_train, y_train, X_val, y_val,
                n_trials=args.tune_trials, sample_weight=train_weights,
            )
            cat_model = cb.CatBoostClassifier(
                iterations=3000,
                random_seed=42,
                verbose=100,
                early_stopping_rounds=50,
                eval_metric="Logloss",
                **cat_best_params,
            )
        else:
            cat_model = cb.CatBoostClassifier(
                iterations=2000,
                depth=6,
                learning_rate=0.01,
                l2_leaf_reg=3.0,
                subsample=0.8,
                random_seed=42,
                verbose=100,
                early_stopping_rounds=50,
                eval_metric="Logloss",
            )
        cat_model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=(X_val, y_val),
        )

        cat_pred = cat_model.predict(X_test)
        cat_prob = cat_model.predict_proba(X_test)[:, 1]
        cat_acc = evaluate(y_test, cat_pred.astype(int), cat_prob,
                           f"CatBoost - {args.interval} Test")

        cat_val_pred = cat_model.predict(X_val)
        cat_val_prob = cat_model.predict_proba(X_val)[:, 1]
        evaluate(y_val, cat_val_pred.astype(int), cat_val_prob,
                 f"CatBoost - {args.interval} Val")
        results["CatBoost"] = (cat_acc, cat_prob, cat_val_prob)
    else:
        print("\n  CatBoost not installed. Skipping. (pip install catboost)")

    # ==================================================================
    # 6.7. Random Forest
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Training Random Forest...")
    print("=" * 60)

    rf_model = RandomForestClassifier(
        n_estimators=600,
        max_depth=12,
        min_samples_leaf=25,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf_model.fit(X_train, y_train, sample_weight=train_weights)

    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_acc = evaluate(y_test, rf_pred, rf_prob, f"Random Forest - {args.interval} Test")

    rf_val_pred = rf_model.predict(X_val)
    rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
    evaluate(y_val, rf_val_pred, rf_val_prob, f"Random Forest - {args.interval} Val")
    results["RandomForest"] = (rf_acc, rf_prob, rf_val_prob)

    # ==================================================================
    # 7. Ensemble (average probabilities + optimized weighted blend)
    # ==================================================================
    ensemble_parts = [
        ("LogisticRegression", lr_val_prob, lr_prob),
        ("XGBoost", xgb_val_prob, xgb_prob),
        ("LightGBM", lgb_val_prob, lgb_prob),
        ("RandomForest", rf_val_prob, rf_prob),
    ]
    if cat_val_prob is not None:
        ensemble_parts.append(("CatBoost", cat_val_prob, cat_prob))

    base_model_names = [name for name, _, _ in ensemble_parts]
    base_val_probs = [val_p for _, val_p, _ in ensemble_parts]
    base_test_probs = [test_p for _, _, test_p in ensemble_parts]

    blend_val_X = np.column_stack(base_val_probs)
    blend_test_X = np.column_stack(base_test_probs)

    # ------------------------------------------------------------------
    # 7.1 Equal-weight ensemble
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Ensemble (Average Probabilities)")
    print("=" * 60)

    ensemble_val_prob = clip_probs(blend_val_X.mean(axis=1))
    ensemble_prob = clip_probs(blend_test_X.mean(axis=1))
    ens_label = f"Ensemble Equal ({'+'.join(base_model_names)})"

    ensemble_pred = (ensemble_prob > 0.5).astype(int)
    ens_acc = evaluate(y_test, ensemble_pred, ensemble_prob,
                       f"{ens_label} - {args.interval} Test")

    ensemble_val_pred = (ensemble_val_prob > 0.5).astype(int)
    evaluate(y_val, ensemble_val_pred, ensemble_val_prob,
             f"{ens_label} - {args.interval} Val")
    results["Ensemble"] = (ens_acc, ensemble_prob, ensemble_val_prob)

    # ------------------------------------------------------------------
    # 7.2 Weighted ensemble (weights tuned on val log-loss)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Weighted Ensemble (optimize val log-loss)")
    print("=" * 60)

    ens_weights, best_val_ll = optimize_ensemble_weights(blend_val_X, y_val, n_iter=4000, seed=42)
    weighted_val_prob = clip_probs(blend_val_X @ ens_weights)
    weighted_test_prob = clip_probs(blend_test_X @ ens_weights)

    weighted_test_pred = (weighted_test_prob > 0.5).astype(int)
    weighted_acc = evaluate(
        y_test, weighted_test_pred, weighted_test_prob,
        f"Weighted Ensemble - {args.interval} Test"
    )

    weighted_val_pred = (weighted_val_prob > 0.5).astype(int)
    evaluate(
        y_val, weighted_val_pred, weighted_val_prob,
        f"Weighted Ensemble - {args.interval} Val"
    )
    results["WeightedEnsemble"] = (weighted_acc, weighted_test_prob, weighted_val_prob)

    print(f"  Best val log-loss (weighted ensemble): {best_val_ll:.5f}")
    print("  Ensemble weights:")
    for name, w in zip(base_model_names, ens_weights):
        print(f"    {name:<20s} {w:.4f}")

    # ==================================================================
    # 7.5. Stacking Ensemble (meta-learner on base model predictions)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Stacking Ensemble (meta-learner)")
    print("=" * 60)

    meta_scaler = StandardScaler()
    meta_val_X_s = meta_scaler.fit_transform(blend_val_X)
    meta_test_X_s = meta_scaler.transform(blend_test_X)

    meta_model = LogisticRegression(C=0.2, max_iter=2000, solver="lbfgs", random_state=42)
    meta_model.fit(meta_val_X_s, y_val)

    stack_prob = clip_probs(meta_model.predict_proba(meta_test_X_s)[:, 1])
    stack_pred = (stack_prob > 0.5).astype(int)
    stack_acc = evaluate(y_test, stack_pred, stack_prob,
                         f"Stacked Ensemble - {args.interval} Test")

    # For threshold analysis val predictions, use cross-validated stacking on val set
    from sklearn.model_selection import TimeSeriesSplit
    stack_val_prob = np.full(len(y_val), np.nan, dtype=np.float64)
    cv = TimeSeriesSplit(n_splits=5)
    for tr_idx, te_idx in cv.split(blend_val_X):
        fold_scaler = StandardScaler()
        fold_X_tr = fold_scaler.fit_transform(blend_val_X[tr_idx])
        fold_X_te = fold_scaler.transform(blend_val_X[te_idx])
        fold_meta = LogisticRegression(C=0.2, max_iter=2000, solver="lbfgs", random_state=42)
        fold_meta.fit(fold_X_tr, y_val[tr_idx])
        stack_val_prob[te_idx] = fold_meta.predict_proba(fold_X_te)[:, 1]

    # TimeSeriesSplit does not score the earliest prefix; backfill with
    # full-val meta predictions so threshold analysis has complete coverage.
    if np.isnan(stack_val_prob).any():
        fallback_val_prob = meta_model.predict_proba(meta_val_X_s)[:, 1]
        nan_mask = np.isnan(stack_val_prob)
        stack_val_prob[nan_mask] = fallback_val_prob[nan_mask]

    stack_val_prob = clip_probs(stack_val_prob)
    stack_val_pred = (stack_val_prob > 0.5).astype(int)
    evaluate(y_val, stack_val_pred, stack_val_prob,
             f"Stacked Ensemble - {args.interval} Val")
    results["Stacked"] = (stack_acc, stack_prob, stack_val_prob)

    # ==================================================================
    # 8. Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"  SUMMARY — {args.interval} Test Set")
    print("=" * 60)
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'AUC-ROC':>10s} {'LogLoss':>10s} {'Brier':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, (acc, prob, _) in results.items():
        try:
            prob_clip = clip_probs(prob)
            auc = roc_auc_score(y_test, prob_clip)
            ll = log_loss(y_test, prob_clip)
            brier = brier_score_loss(y_test, prob_clip)
        except Exception:
            auc = float("nan")
            ll = float("nan")
            brier = float("nan")
        print(f"  {name:<25s} {acc:>10.4f} {auc:>10.4f} {ll:>10.4f} {brier:>10.4f}")
    print(f"  {'CryptoMamba (ref)':<25s} {'~0.51':>10s} {'—':>10s} {'—':>10s} {'—':>10s}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 9. Feature importance (top 20)
    # ------------------------------------------------------------------
    print(f"\n  Top 20 Features — XGBoost (gain):")
    xgb_imp = sorted(zip(feature_cols, xgb_model.feature_importances_),
                      key=lambda x: x[1], reverse=True)
    for name, imp in xgb_imp[:20]:
        print(f"    {name:<30s}  {imp:.4f}")

    print(f"\n  Top 20 Features — LightGBM (split count):")
    lgb_imp = sorted(zip(feature_cols, lgb_model.feature_importances_),
                      key=lambda x: x[1], reverse=True)
    for name, imp in lgb_imp[:20]:
        print(f"    {name:<30s}  {imp:.0f}")

    if HAS_CATBOOST:
        print(f"\n  Top 20 Features — CatBoost:")
        cat_imp_vals = cat_model.get_feature_importance()
        cat_imp = sorted(zip(feature_cols, cat_imp_vals),
                          key=lambda x: x[1], reverse=True)
        for name, imp in cat_imp[:20]:
            print(f"    {name:<30s}  {imp:.2f}")

    print(f"\n  Top 20 Features — RandomForest (impurity):")
    rf_imp = sorted(zip(feature_cols, rf_model.feature_importances_),
                    key=lambda x: x[1], reverse=True)
    for name, imp in rf_imp[:20]:
        print(f"    {name:<30s}  {imp:.4f}")

    # ------------------------------------------------------------------
    # 10. Threshold-based betting analysis
    # ------------------------------------------------------------------
    # Compute number of calendar days in val and test periods
    val_days = (val_e - val_s) / 86400
    test_days = (test_e - test_s) / 86400

    threshold_results = {}  # name -> (val_results, test_results)
    for name, (acc, test_prob, val_prob) in results.items():
        val_res = threshold_analysis(y_val, val_prob, name, val_days)
        test_res = threshold_analysis(y_test, test_prob, name, test_days)
        threshold_results[name] = (val_res, test_res)

    print_threshold_table(threshold_results, val_days, test_days)

    # ------------------------------------------------------------------
    # 11. Save models (optional)
    # ------------------------------------------------------------------
    if args.save_models:
        save_dir = f"{ROOT}/checkpoints/baselines/{args.interval}"
        os.makedirs(save_dir, exist_ok=True)

        xgb_model.save_model(f"{save_dir}/xgboost.json")
        lgb_model.booster_.save_model(f"{save_dir}/lightgbm.txt")
        if HAS_CATBOOST:
            cat_model.save_model(f"{save_dir}/catboost.cbm")
        with open(f"{save_dir}/random_forest.pkl", "wb") as f:
            pickle.dump(rf_model, f)
        with open(f"{save_dir}/logistic_regression.pkl", "wb") as f:
            pickle.dump({"model": lr_model, "scaler": scaler}, f)
        with open(f"{save_dir}/ensemble_weights.pkl", "wb") as f:
            pickle.dump({
                "base_model_names": base_model_names,
                "weights": ens_weights,
                "objective": "val_logloss_minimized",
            }, f)
        with open(f"{save_dir}/feature_cols.pkl", "wb") as f:
            pickle.dump(feature_cols, f)

        print(f"\n  Models saved to {save_dir}/")
