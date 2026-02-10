"""
Threshold analysis for baseline models.

Trains XGBoost, LightGBM, Logistic Regression, and Ensemble, then sweeps
across confidence thresholds to find the optimal betting strategy.

Usage:
    python scripts/threshold_analysis.py --interval 15m
    python scripts/threshold_analysis.py --interval 15m --plot
    python scripts/threshold_analysis.py --interval 15m --thresholds 0.50 0.70 0.005
"""

import os
import sys
import pathlib

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from scripts.baseline_models import engineer_features, INTERVAL_CONFIG

ROOT = os.path.dirname(pathlib.Path(__file__).parent.absolute())


# ---------------------------------------------------------------------------
# Polymarket fee model (15-minute crypto markets)
# https://docs.polymarket.com/polymarket-learn/trading/maker-rebates-program
#
#   fee_per_share = share_price × 0.25 × (share_price × (1 - share_price))²
#   fee_rate      = fee_per_share / share_price = 0.25 × (p × (1-p))²
#
#   At p=0.50 → fee_rate = 1.5625%  (maximum, worst case)
#   At p=0.10 → fee_rate = 0.2025%
#   At p=0.90 → fee_rate = 0.2025%
#
#   Fee is charged on purchase only. Shares auto-resolve (no sell fee).
#   Total cost to buy N shares at price p:
#     cost = N × p × (1 + fee_rate)
#   If win (share pays $1):
#     profit = N × 1.0 - cost = cost × (1/(p × (1+fee_rate)) - 1)
#   If lose (share pays $0):
#     loss = -cost
# ---------------------------------------------------------------------------
def polymarket_fee_rate(share_price):
    """Polymarket taker fee rate as fraction of share cost.

    fee_rate = 0.25 × (p × (1-p))²
    Returns fee as fraction (e.g., 0.015625 at p=0.50).
    """
    p = share_price
    return 0.25 * (p * (1.0 - p)) ** 2


def get_args():
    parser = ArgumentParser(description="Threshold sweep analysis for baseline models")
    parser.add_argument("--interval", type=str, default="15m",
                        choices=["15m", "30m", "1h"])
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Save accuracy-vs-threshold plot")
    parser.add_argument("--thresholds", nargs=3, type=float, default=None,
                        metavar=("START", "STOP", "STEP"),
                        help="Threshold range: start stop step (default: 0.50 0.70 0.005)")
    parser.add_argument("--market_price", type=float, default=0.50,
                        help="Polymarket YES/UP price (default: 0.50 = fair odds)")
    parser.add_argument("--use_weights", action="store_true", default=False,
                        help="Enable sample weighting by return magnitude")
    return parser.parse_args()


def sweep_thresholds(y_true, y_prob, thresholds, num_days):
    """Sweep thresholds and compute metrics for each.

    Returns a DataFrame with columns:
        threshold, accuracy, precision, recall, f1, num_bets, bets_per_day, edge
    """
    rows = []
    for t in thresholds:
        confidence = np.maximum(y_prob, 1.0 - y_prob)
        mask = confidence > t
        n = mask.sum()

        if n < 5:
            rows.append({
                "threshold": t, "accuracy": np.nan, "precision": np.nan,
                "recall": np.nan, "f1": np.nan,
                "num_bets": int(n), "bets_per_day": n / max(num_days, 1),
                "edge": np.nan,
            })
            continue

        y_sel = y_true[mask]
        pred_sel = (y_prob[mask] > 0.5).astype(int)

        acc = accuracy_score(y_sel, pred_sel)
        prec = precision_score(y_sel, pred_sel, zero_division=0)
        rec = recall_score(y_sel, pred_sel, zero_division=0)
        f1 = f1_score(y_sel, pred_sel, zero_division=0)

        rows.append({
            "threshold": t,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "num_bets": int(n),
            "bets_per_day": n / max(num_days, 1),
            "edge": (acc - 0.5) * 100,
        })

    return pd.DataFrame(rows)


def train_models(X_train, y_train, X_val, y_val, train_weights=None):
    """Train all models and return dict of {name: model}."""
    models = {}

    # Logistic Regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    lr.fit(X_train_s, y_train)
    models["LR"] = {"model": lr, "scaler": scaler}

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000, max_depth=6, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric="logloss", early_stopping_rounds=50,
    )
    xgb_model.fit(X_train, y_train, sample_weight=train_weights,
                   eval_set=[(X_val, y_val)], verbose=0)
    models["XGBoost"] = {"model": xgb_model}

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000, max_depth=6, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train, sample_weight=train_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    models["LightGBM"] = {"model": lgb_model}

    # CatBoost
    if HAS_CATBOOST:
        cat_model = cb.CatBoostClassifier(
            iterations=2000, depth=6, learning_rate=0.01,
            l2_leaf_reg=3.0, subsample=0.8,
            random_seed=42, verbose=0,
            early_stopping_rounds=50, eval_metric="Logloss",
        )
        cat_model.fit(X_train, y_train, sample_weight=train_weights,
                       eval_set=(X_val, y_val))
        models["CatBoost"] = {"model": cat_model}

    return models


def predict_proba(models, X, scaler=None):
    """Get P(UP) from each model + ensemble."""
    probs = {}
    for name, m in models.items():
        if name == "LR":
            X_in = m["scaler"].transform(X)
        else:
            X_in = X
        probs[name] = m["model"].predict_proba(X_in)[:, 1]

    probs["Ensemble"] = np.mean(list(probs.values()), axis=0)
    return probs


def print_sweep_table(df, model_name, split_name):
    """Print a single model's sweep results."""
    print(f"\n  {model_name} — {split_name}")
    print(f"  {'Thresh':>6s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} "
          f"{'F1':>7s} {'Bets':>7s} {'Bets/D':>7s} {'Edge':>7s}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for _, r in df.iterrows():
        def fmt(v, spec=".4f"):
            return f"{v:{spec}}" if not np.isnan(v) else "   N/A"

        print(f"  {r['threshold']:>6.3f} {fmt(r['accuracy']):>7s} "
              f"{fmt(r['precision']):>7s} {fmt(r['recall']):>7s} "
              f"{fmt(r['f1']):>7s} {int(r['num_bets']):>7d} "
              f"{r['bets_per_day']:>7.1f} {fmt(r['edge'], '+.1f'):>7s}")


def find_best_thresholds(val_dfs, test_dfs, model_names):
    """Find optimal threshold on val for each model, report test performance."""
    bpd_ranges = [
        ("Conservative  (5-15/day)", 5.0, 15.0),
        ("Moderate     (10-30/day)", 10.0, 30.0),
        ("Aggressive   (15-50/day)", 15.0, 50.0),
    ]

    print(f"\n{'=' * 88}")
    print(f"  BEST THRESHOLDS (selected on val accuracy, applied to test)")
    print(f"{'=' * 88}")

    for style, lo, hi in bpd_ranges:
        print(f"\n  {style}:")
        print(f"  {'Model':<12s} {'Thresh':>6s} {'ValAcc':>7s} {'TestAcc':>8s} "
              f"{'TestPrec':>9s} {'TestRec':>8s} {'Bets':>6s} {'B/Day':>6s} {'Edge':>7s}")
        print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*8} {'-'*9} {'-'*8} "
              f"{'-'*6} {'-'*6} {'-'*7}")

        for name in model_names:
            vdf = val_dfs[name]
            tdf = test_dfs[name]

            # Filter to target bets/day range on val
            candidates = vdf[(vdf["bets_per_day"] >= lo) &
                             (vdf["bets_per_day"] <= hi) &
                             (vdf["accuracy"].notna())]

            if candidates.empty:
                print(f"  {name:<12s}   (no threshold in range)")
                continue

            best_idx = candidates["accuracy"].idxmax()
            best_t = candidates.loc[best_idx, "threshold"]

            # Look up same threshold in test
            test_row = tdf[tdf["threshold"] == best_t].iloc[0]
            val_row = candidates.loc[best_idx]

            def fmt(v, spec=".4f"):
                return f"{v:{spec}}" if not np.isnan(v) else "N/A"

            print(f"  {name:<12s} {best_t:>6.3f} {fmt(val_row['accuracy']):>7s} "
                  f"{fmt(test_row['accuracy']):>8s} "
                  f"{fmt(test_row['precision']):>9s} {fmt(test_row['recall']):>8s} "
                  f"{int(test_row['num_bets']):>6d} {test_row['bets_per_day']:>6.1f} "
                  f"{fmt(test_row['edge'], '+.1f'):>7s}")

    print(f"\n{'=' * 88}")


def save_plot(val_dfs, test_dfs, model_names, interval, out_path):
    """Save accuracy-vs-threshold and bets/day-vs-threshold plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"LR": "#7f8c8d", "XGBoost": "#e74c3c",
              "LightGBM": "#2ecc71", "CatBoost": "#9b59b6",
              "Ensemble": "#3498db", "Stacked": "#e67e22"}

    # --- Left: Accuracy vs Threshold ---
    ax = axes[0]
    for name in model_names:
        df = test_dfs[name]
        valid = df[df["accuracy"].notna()]
        ax.plot(valid["threshold"], valid["accuracy"] * 100,
                marker="o", markersize=3, label=name,
                color=colors.get(name, None), linewidth=2)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(f"BTC {interval} — Accuracy vs Confidence Threshold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.50, 0.65)

    # --- Right: Bets per Day vs Threshold ---
    ax = axes[1]
    for name in model_names:
        df = test_dfs[name]
        valid = df[df["accuracy"].notna()]
        ax.plot(valid["threshold"], valid["bets_per_day"],
                marker="o", markersize=3, label=name,
                color=colors.get(name, None), linewidth=2)
    ax.axhline(y=10, color="orange", linestyle="--", alpha=0.5, label="10 bets/day")
    ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="20 bets/day")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Bets per Day", fontsize=12)
    ax.set_title(f"BTC {interval} — Bet Frequency vs Confidence Threshold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.50, 0.65)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n  Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Kelly Criterion Betting Simulation
# ---------------------------------------------------------------------------
def kelly_simulation(y_true, y_prob, threshold, market_price=0.50,
                     kelly_fraction=0.25, num_days=127, include_fees=True):
    """Simulate bankroll growth using Kelly Criterion on filtered bets.

    Polymarket model:
      - Buy YES share at price `m`, pays $1 if correct, $0 if wrong.
      - Buy NO  share at price `1-m`, pays $1 if correct, $0 if wrong.
      - Taker fee: fee_rate = 0.25 × (share_price × (1-share_price))²
      - Fee is on purchase only; shares auto-resolve at expiry.

    Kelly fraction for YES bet: f* = (p - m) / (1 - m)
    Kelly fraction for NO  bet: f* = (m - p) / m
    (adjusted for fees in the profit calculation)

    Only bet when:
      1) confidence > threshold
      2) Kelly fraction > 0  (model disagrees with market)

    Args:
        y_true:         actual labels (0=DOWN, 1=UP)
        y_prob:         model's predicted P(UP)
        threshold:      minimum confidence to place a bet
        market_price:   Polymarket price for YES/UP (default 0.50)
        kelly_fraction: fraction of full Kelly to use (e.g. 0.25 = quarter Kelly)
        num_days:       number of calendar days for annualization
        include_fees:   whether to include Polymarket taker fees

    Returns:
        dict with simulation results
    """
    bankroll = 1.0  # start with $1 (everything in fractions)
    initial_bankroll = bankroll
    bankroll_history = [bankroll]

    n_bets = 0
    n_wins = 0
    n_skip_confidence = 0
    n_skip_no_edge = 0
    max_bankroll = bankroll
    max_drawdown = 0.0
    total_wagered = 0.0
    total_fees = 0.0
    bet_details = []

    m = market_price

    for i in range(len(y_true)):
        p = y_prob[i]
        actual = y_true[i]

        # Step 1: Check confidence threshold
        confidence = max(p, 1.0 - p)
        if confidence <= threshold:
            n_skip_confidence += 1
            bankroll_history.append(bankroll)
            continue

        # Step 2: Determine bet direction and Kelly sizing
        if p > m:
            # Model says UP is more likely than market thinks -> buy YES
            full_kelly = (p - m) / (1.0 - m)
            direction = "UP"
            win = (actual == 1)
            share_price = m
        elif p < (1.0 - m):
            # Model says DOWN is more likely -> buy NO
            full_kelly = (m - p) / m
            direction = "DOWN"
            win = (actual == 0)
            share_price = 1.0 - m
        else:
            # Model agrees with market, no edge
            n_skip_no_edge += 1
            bankroll_history.append(bankroll)
            continue

        # Step 3: Apply fractional Kelly
        f = full_kelly * kelly_fraction
        f = min(f, 0.05)  # cap at 5% of bankroll per bet for safety
        f = max(f, 0.0)

        if f < 0.001:  # too small to bother
            n_skip_no_edge += 1
            bankroll_history.append(bankroll)
            continue

        # Step 4: Execute bet (with Polymarket fees)
        bet_amount = f * bankroll
        total_wagered += bet_amount
        n_bets += 1

        if include_fees:
            fee_rate = polymarket_fee_rate(share_price)
            fee = bet_amount * fee_rate
        else:
            fee_rate = 0.0
            fee = 0.0
        total_fees += fee

        # Shares purchased: (bet_amount - fee) / share_price
        # Total out-of-pocket = bet_amount (fee comes out of the bet budget)
        effective_spend = bet_amount - fee
        shares = effective_spend / share_price

        if win:
            profit = shares * 1.0 - bet_amount  # payout - total cost
            n_wins += 1
        else:
            profit = -bet_amount

        bankroll += profit
        bankroll_history.append(bankroll)

        # Track drawdown
        if bankroll > max_bankroll:
            max_bankroll = bankroll
        dd = (max_bankroll - bankroll) / max_bankroll
        if dd > max_drawdown:
            max_drawdown = dd

        bet_details.append({
            "direction": direction, "prob": p, "kelly_f": f,
            "bet_amount": bet_amount, "win": win, "profit": profit,
            "fee": fee, "bankroll": bankroll,
        })

    # Compute summary stats
    total_return = (bankroll / initial_bankroll - 1.0) * 100
    win_rate = n_wins / n_bets if n_bets > 0 else 0
    bets_per_day = n_bets / max(num_days, 1)
    avg_bet_size = total_wagered / n_bets if n_bets > 0 else 0

    # Profit factor
    gross_profit = sum(d["profit"] for d in bet_details if d["profit"] > 0)
    gross_loss = abs(sum(d["profit"] for d in bet_details if d["profit"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "bankroll_final": bankroll,
        "total_return_pct": total_return,
        "n_bets": n_bets,
        "n_wins": n_wins,
        "win_rate": win_rate,
        "bets_per_day": bets_per_day,
        "avg_bet_size_pct": avg_bet_size * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "profit_factor": profit_factor,
        "total_wagered": total_wagered,
        "total_fees": total_fees,
        "fee_pct_of_wagered": total_fees / total_wagered * 100 if total_wagered > 0 else 0,
        "n_skip_confidence": n_skip_confidence,
        "n_skip_no_edge": n_skip_no_edge,
        "bankroll_history": bankroll_history,
        "bet_details": bet_details,
    }


def flat_bet_analysis(y_true, y_prob, threshold, market_price, num_days,
                      include_fees=True):
    """Non-compounding flat-bet analysis: bet $1 per trade.

    This is more realistic for Polymarket where each market is independent.
    Includes Polymarket taker fees when include_fees=True.

    Returns dict with flat-bet metrics.
    """
    m = market_price
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    mask = confidence > threshold
    n = mask.sum()

    if n < 5:
        return None

    y_sel = y_true[mask]
    p_sel = y_prob[mask]

    total_profit = 0.0
    total_wagered = 0.0
    total_fees = 0.0
    n_bets = 0
    n_wins = 0
    profits = []

    for i in range(len(y_sel)):
        p = p_sel[i]
        actual = y_sel[i]
        bet = 1.0  # flat $1 per trade

        if p > m:
            share_price = m
            won = (actual == 1)
        else:
            share_price = 1.0 - m
            won = (actual == 0)

        # Apply Polymarket taker fee
        if include_fees:
            fee_rate = polymarket_fee_rate(share_price)
            fee = bet * fee_rate
        else:
            fee = 0.0
        total_fees += fee

        # Shares = (bet - fee) / share_price
        effective_spend = bet - fee
        shares = effective_spend / share_price

        if won:
            profit = shares * 1.0 - bet  # payout minus total cost
            n_wins += 1
        else:
            profit = -bet

        total_profit += profit
        total_wagered += bet
        n_bets += 1
        profits.append(profit)

    profits = np.array(profits)
    roi = total_profit / total_wagered if total_wagered > 0 else 0
    win_rate = n_wins / n_bets if n_bets > 0 else 0
    bets_per_day = n_bets / max(num_days, 1)
    daily_profit = total_profit / max(num_days, 1)

    # Avg Kelly fraction (what Kelly RECOMMENDS as bet size)
    kelly_fracs = []
    for i in range(len(y_sel)):
        p = p_sel[i]
        if p > m:
            kf = (p - m) / (1.0 - m)
        else:
            kf = (m - p) / m
        kelly_fracs.append(kf)
    avg_kelly = np.mean(kelly_fracs) if kelly_fracs else 0

    return {
        "n_bets": n_bets,
        "n_wins": n_wins,
        "win_rate": win_rate,
        "bets_per_day": bets_per_day,
        "total_wagered": total_wagered,
        "total_profit": total_profit,
        "total_fees": total_fees,
        "roi_pct": roi * 100,
        "daily_profit": daily_profit,
        "avg_profit_per_bet": total_profit / n_bets if n_bets > 0 else 0,
        "profit_std": profits.std() if len(profits) > 1 else 0,
        "avg_kelly_frac": avg_kelly,
    }


def print_flat_bet_results(y_true, probs_dict, model_names,
                           thresholds, market_price, num_days):
    """Print flat-bet analysis with fee impact comparison."""
    m = market_price
    fee_at_50 = polymarket_fee_rate(0.50) * 100

    print(f"\n{'=' * 100}")
    print(f"  FLAT-BET ANALYSIS (no compounding, $1 per bet)")
    print(f"  Market price = {m:.2f}  |  Test period = {num_days:.0f} days")
    print(f"  Polymarket taker fee at p=0.50: {fee_at_50:.3f}%")
    print(f"{'=' * 100}")

    print(f"\n  Kelly formula for Polymarket:")
    print(f"    Bet YES: f* = (p - m) / (1 - m)     when model P(UP) = p > m")
    print(f"    Bet NO:  f* = (m - p) / m            when model P(UP) = p < 1-m")
    print(f"    Practical: use 1/4 Kelly (safer), cap at 5% of bankroll")

    # Compute both with and without fees
    rows = []
    for name in model_names:
        for t in thresholds:
            res_fee = flat_bet_analysis(
                y_true, probs_dict[name], t, market_price, num_days,
                include_fees=True)
            res_nofee = flat_bet_analysis(
                y_true, probs_dict[name], t, market_price, num_days,
                include_fees=False)
            if res_fee is None or res_fee["n_bets"] < 10:
                continue
            rows.append({
                "model": name, "threshold": t,
                **{f"{k}": v for k, v in res_fee.items()},
                "roi_nofee_pct": res_nofee["roi_pct"],
                "profit_nofee": res_nofee["total_profit"],
            })

    rows.sort(key=lambda r: r["roi_pct"], reverse=True)

    # Full table
    print(f"\n  {'Model':<12s} {'Thresh':>6s} {'WinRate':>7s} {'Bets':>6s} "
          f"{'B/Day':>5s} {'ROI':>8s} {'ROI*':>8s} {'Fee$':>7s} "
          f"{'$Total':>8s} {'$/Day':>7s}")
    print(f"  {' '*12} {' '*6} {' '*7} {' '*6} "
          f"{' '*5} {'w/fee':>8s} {'no fee':>8s} {'paid':>7s} "
          f"{'w/fee':>8s} {'w/fee':>7s}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*6} {'-'*5} "
          f"{'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")

    for r in rows:
        print(f"  {r['model']:<12s} {r['threshold']:>6.3f} {r['win_rate']:>6.1%} "
              f"{r['n_bets']:>6d} {r['bets_per_day']:>5.1f} "
              f"{r['roi_pct']:>+7.1f}% "
              f"{r['roi_nofee_pct']:>+7.1f}% "
              f"{r['total_fees']:>7.1f} "
              f"{r['total_profit']:>+8.1f} "
              f"{r['daily_profit']:>+7.2f}")

    # Best per model
    print(f"\n  {'=' * 95}")
    print(f"  BEST STRATEGY PER MODEL (by ROI after fees, min 5 bets/day)")
    print(f"  {'=' * 95}")

    print(f"  {'Model':<12s} {'Thresh':>6s} {'WinRate':>7s} {'B/Day':>5s} "
          f"{'ROI':>8s} {'ROI*':>8s} {'FeeCost':>8s} {'$/Bet':>7s} "
          f"{'$/Day':>7s} {'1/4 K%':>7s}")
    print(f"  {' '*12} {' '*6} {' '*7} {' '*5} "
          f"{'w/fee':>8s} {'no fee':>8s} {'':>8s} {'w/fee':>7s} "
          f"{'w/fee':>7s} {'':>7s}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*5} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    for name in model_names:
        model_rows = [r for r in rows
                      if r["model"] == name and r["bets_per_day"] >= 5]
        if not model_rows:
            continue
        best = max(model_rows, key=lambda r: r["roi_pct"])
        qk = best["avg_kelly_frac"] * 0.25
        fee_cost = f"{best['total_fees'] / best['total_wagered'] * 100:.2f}%"
        print(f"  {name:<12s} {best['threshold']:>6.3f} {best['win_rate']:>6.1%} "
              f"{best['bets_per_day']:>5.1f} "
              f"{best['roi_pct']:>+7.1f}% "
              f"{best['roi_nofee_pct']:>+7.1f}% "
              f"{fee_cost:>8s} "
              f"{best['avg_profit_per_bet']:>+7.3f} "
              f"{best['daily_profit']:>+7.2f} "
              f"{qk:>6.1%}")

    print(f"\n  Interpretation:")
    print(f"    ROI w/fee  = net profit / total wagered (after Polymarket fees)")
    print(f"    ROI* no fee = profit if no fees (for comparison)")
    print(f"    FeeCost    = total fees as % of wagered")
    print(f"    $/Bet      = avg net profit per $1 bet (after fees)")
    print(f"    $/Day      = avg daily net profit per $1/trade")
    print(f"    1/4 K%     = recommended bet size (quarter Kelly)")
    print(f"    To scale: $/Day × bet_size = daily profit")
    print(f"    Example: $100/bet, $/Day=+2.00 → daily profit ~ $200")
    print(f"  {'=' * 95}")


def save_kelly_plot(y_true, probs_dict, best_configs, market_price,
                    num_days, interval, out_path):
    """Plot bankroll curves for the best strategies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"LR": "#7f8c8d", "XGBoost": "#e74c3c",
              "LightGBM": "#2ecc71", "CatBoost": "#9b59b6",
              "Ensemble": "#3498db", "Stacked": "#e67e22"}

    for name, threshold, kf in best_configs:
        sim = kelly_simulation(
            y_true, probs_dict[name], threshold=threshold,
            market_price=market_price, kelly_fraction=kf, num_days=num_days,
        )
        history = sim["bankroll_history"]
        # x-axis: bet index (approximate time progression)
        x = np.arange(len(history))
        label = (f"{name} t={threshold:.2f} {kf:.0%}K "
                 f"(ret={sim['total_return_pct']:+.1f}%)")
        ax.plot(x, history, label=label,
                color=colors.get(name, None), linewidth=1.5)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Break even")
    ax.set_xlabel("Sample index (chronological)", fontsize=12)
    ax.set_ylabel("Bankroll ($)", fontsize=12)
    ax.set_title(f"BTC {interval} — Kelly Criterion Bankroll Simulation", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Bankroll plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    config = INTERVAL_CONFIG[args.interval]
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # Threshold range
    if args.thresholds:
        t_start, t_stop, t_step = args.thresholds
    else:
        t_start, t_stop, t_step = 0.500, 0.700, 0.005
    thresholds = np.arange(t_start, t_stop + 1e-9, t_step)

    # ------------------------------------------------------------------
    # 1. Load & prepare data
    # ------------------------------------------------------------------
    print(f"Loading {config['data_path']}...")
    df = pd.read_csv(config["data_path"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    print("Engineering features...")
    features = engineer_features(df, base_interval_seconds=config["jumps"])
    features = features.dropna()
    feature_cols = [c for c in features.columns if c != "target" and not c.startswith("_")]
    print(f"  {len(feature_cols)} features, {len(features)} samples")

    # Split
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

    val_days = (val_e - val_s) / 86400
    test_days = (test_e - test_s) / 86400

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} ({val_days:.0f} days) "
          f"| Test: {len(X_test)} ({test_days:.0f} days)")

    # Compute sample weights (optional)
    if args.use_weights:
        return_mag_train = features.loc[features.index[train_mask], "_return_mag"].values
        train_weights = np.clip(return_mag_train / 0.001, 0.2, 5.0)
        train_weights = train_weights / train_weights.mean()
        print(f"  Sample weights: ON")
    else:
        train_weights = None
        print(f"  Sample weights: OFF")

    # ------------------------------------------------------------------
    # 2. Train models
    # ------------------------------------------------------------------
    print("\nTraining models...")
    models = train_models(X_train, y_train, X_val, y_val, train_weights=train_weights)
    print("  Done.\n")

    # ------------------------------------------------------------------
    # 3. Get probabilities
    # ------------------------------------------------------------------
    val_probs = predict_proba(models, X_val)
    test_probs = predict_proba(models, X_test)

    # Add stacking ensemble (meta-learner on base model predictions)
    from sklearn.model_selection import KFold
    base_names = [n for n in val_probs if n != "Ensemble"]
    meta_val_X = np.column_stack([val_probs[n] for n in base_names])
    meta_test_X = np.column_stack([test_probs[n] for n in base_names])

    meta_scaler = StandardScaler()
    meta_val_X_s = meta_scaler.fit_transform(meta_val_X)
    meta_test_X_s = meta_scaler.transform(meta_test_X)

    meta_model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    meta_model.fit(meta_val_X_s, y_val)
    test_probs["Stacked"] = meta_model.predict_proba(meta_test_X_s)[:, 1]

    # Cross-validated stacking for val predictions
    stack_val_prob = np.zeros(len(y_val))
    kf = KFold(n_splits=5, shuffle=False)
    for tr_idx, te_idx in kf.split(meta_val_X):
        fold_scaler = StandardScaler()
        fold_X_tr = fold_scaler.fit_transform(meta_val_X[tr_idx])
        fold_X_te = fold_scaler.transform(meta_val_X[te_idx])
        fold_meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
        fold_meta.fit(fold_X_tr, y_val[tr_idx])
        stack_val_prob[te_idx] = fold_meta.predict_proba(fold_X_te)[:, 1]
    val_probs["Stacked"] = stack_val_prob

    model_names = list(val_probs.keys())

    # ------------------------------------------------------------------
    # 4. Sweep thresholds
    # ------------------------------------------------------------------
    val_dfs = {}
    test_dfs = {}
    for name in model_names:
        val_dfs[name] = sweep_thresholds(y_val, val_probs[name], thresholds, val_days)
        test_dfs[name] = sweep_thresholds(y_test, test_probs[name], thresholds, test_days)

    # ------------------------------------------------------------------
    # 5. Print full sweep tables
    # ------------------------------------------------------------------
    print(f"{'=' * 70}")
    print(f"  THRESHOLD SWEEP — BTC {args.interval}")
    print(f"  Thresholds: {t_start:.3f} to {t_stop:.3f} step {t_step:.3f}")
    print(f"{'=' * 70}")

    for name in model_names:
        print_sweep_table(test_dfs[name], name, "Test")

    # ------------------------------------------------------------------
    # 6. Best thresholds summary
    # ------------------------------------------------------------------
    find_best_thresholds(val_dfs, test_dfs, model_names)

    # ------------------------------------------------------------------
    # 7. Flat-Bet Analysis + Kelly Sizing
    # ------------------------------------------------------------------
    kelly_thresholds = [0.53, 0.535, 0.54, 0.545, 0.55, 0.555, 0.56, 0.57, 0.58]
    print_flat_bet_results(
        y_test, test_probs, model_names,
        thresholds=kelly_thresholds,
        market_price=args.market_price,
        num_days=test_days,
    )

    # Also run compounding Kelly simulation — compare with/without fees
    kelly_frac = 0.05
    fee_at_50 = polymarket_fee_rate(0.50) * 100

    print(f"\n{'=' * 100}")
    print(f"  COMPOUNDING SIMULATION (5% Kelly, max 5% bankroll/bet)")
    print(f"  Polymarket taker fee at p=0.50: {fee_at_50:.3f}%")
    print(f"{'=' * 100}")
    print(f"  {'Model':<12s} {'Thresh':>6s} {'Final$':>8s} {'Return':>10s} "
          f"{'Ret*':>10s} {'Fees':>8s} {'WinRate':>8s} "
          f"{'Bets':>5s} {'B/Day':>5s} {'MaxDD':>6s}")
    print(f"  {' '*12} {' '*6} {'w/fee':>8s} {'w/fee':>10s} "
          f"{'no fee':>10s} {'total':>8s} {' '*8} "
          f"{' '*5} {' '*5} {'w/fee':>6s}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*10} {'-'*10} "
          f"{'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*6}")

    compound_results = []
    for name in model_names:
        for t in kelly_thresholds:
            sim_fee = kelly_simulation(
                y_test, test_probs[name], threshold=t,
                market_price=args.market_price, kelly_fraction=kelly_frac,
                num_days=test_days, include_fees=True,
            )
            sim_nofee = kelly_simulation(
                y_test, test_probs[name], threshold=t,
                market_price=args.market_price, kelly_fraction=kelly_frac,
                num_days=test_days, include_fees=False,
            )
            if sim_fee["n_bets"] < 10:
                continue
            compound_results.append((name, t, sim_fee, sim_nofee))

    compound_results.sort(key=lambda x: x[2]["total_return_pct"], reverse=True)
    for name, t, sim, sim_nf in compound_results[:20]:
        ret_s = f"{sim['total_return_pct']:+.1f}%"
        ret_nf = f"{sim_nf['total_return_pct']:+.1f}%"
        print(f"  {name:<12s} {t:>6.3f} "
              f"${sim['bankroll_final']:>7.2f} "
              f"{ret_s:>10s} {ret_nf:>10s} "
              f"${sim['total_fees']:>6.2f} "
              f"{sim['win_rate']:>7.1%} "
              f"{sim['n_bets']:>5d} {sim['bets_per_day']:>5.1f} "
              f"{sim['max_drawdown_pct']:>5.1f}%")
    print(f"  {'=' * 100}")

    # ------------------------------------------------------------------
    # 8. Plot (optional)
    # ------------------------------------------------------------------
    if args.plot:
        plot_dir = f"{ROOT}/Results/threshold_analysis"
        os.makedirs(plot_dir, exist_ok=True)
        save_plot(val_dfs, test_dfs, model_names, args.interval,
                  f"{plot_dir}/threshold_{args.interval}.png")

        # Best strategy per model for bankroll plot (5% Kelly)
        best_configs = []
        for name in model_names:
            best_ret = -999
            best_t = 0.55
            for nm, t, sim, _ in compound_results:
                if nm == name and sim["total_return_pct"] > best_ret and sim["bets_per_day"] >= 5:
                    best_ret = sim["total_return_pct"]
                    best_t = t
            if best_ret > -999:
                best_configs.append((name, best_t, kelly_frac))

        if best_configs:
            save_kelly_plot(
                y_test, test_probs, best_configs, args.market_price,
                test_days, args.interval,
                f"{plot_dir}/kelly_bankroll_{args.interval}.png",
            )
