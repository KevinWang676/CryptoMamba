"""
Run XGBoost-only strategy for BTC up/down prediction with confidence thresholding.

Usage:
    python scripts/run_xgb_strategy.py --interval 15m
    python scripts/run_xgb_strategy.py --interval 15m --params_json Results/xgb_best_15m.json
"""

import os
import sys
import json
import pathlib

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

from scripts import baseline_models as bm


def get_args():
    parser = ArgumentParser(description="Run XGBoost strategy with threshold selection")
    parser.add_argument("--interval", type=str, default="15m", choices=["15m", "30m", "1h"])
    parser.add_argument("--params_json", type=str, default=None,
                        help="Optional JSON file containing xgb_params to override defaults")
    parser.add_argument("--use_weights", action="store_true", default=True,
                        help="Use return-magnitude sample weights")
    parser.add_argument("--min_bets_per_day", type=float, default=8.0)
    parser.add_argument("--max_bets_per_day", type=float, default=20.0)
    parser.add_argument("--target_bets_per_day", type=float, default=12.0)
    parser.add_argument("--threshold_min", type=float, default=0.50)
    parser.add_argument("--threshold_max", type=float, default=0.65)
    parser.add_argument("--threshold_step", type=float, default=0.001)
    parser.add_argument("--save_report", type=str, default=None,
                        help="Optional path to save JSON report")
    return parser.parse_args()


def to_ts(s, fmt="%Y-%m-%d %H:%M:%S"):
    return int(datetime.strptime(s, fmt).timestamp())


def load_splits(interval):
    cfg = bm.INTERVAL_CONFIG[interval]
    df = pd.read_csv(cfg["data_path"]).sort_values("Timestamp").reset_index(drop=True)
    feats = bm.engineer_features(df, base_interval_seconds=cfg["jumps"]).dropna()
    feature_cols = [c for c in feats.columns if c != "target" and not c.startswith("_")]

    ts = df["Timestamp"].values[feats.index]
    train_s, train_e = to_ts(cfg["train_interval"][0]), to_ts(cfg["train_interval"][1])
    val_s, val_e = to_ts(cfg["val_interval"][0]), to_ts(cfg["val_interval"][1])
    test_s, test_e = to_ts(cfg["test_interval"][0]), to_ts(cfg["test_interval"][1])

    train_mask = (ts >= train_s) & (ts < train_e)
    val_mask = (ts >= val_s) & (ts < val_e)
    test_mask = (ts >= test_s) & (ts < test_e)

    X_train = feats.loc[feats.index[train_mask], feature_cols].values
    y_train = feats.loc[feats.index[train_mask], "target"].values
    X_val = feats.loc[feats.index[val_mask], feature_cols].values
    y_val = feats.loc[feats.index[val_mask], "target"].values
    X_test = feats.loc[feats.index[test_mask], feature_cols].values
    y_test = feats.loc[feats.index[test_mask], "target"].values

    return_mag_train = feats.loc[feats.index[train_mask], "_return_mag"].values
    val_days = (val_e - val_s) / 86400.0
    test_days = (test_e - test_s) / 86400.0

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "return_mag_train": return_mag_train,
        "feature_cols": feature_cols,
        "val_days": val_days, "test_days": test_days,
    }


def default_xgb_params():
    return {
        "n_estimators": 1400,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.7,
        "min_child_weight": 12,
        "gamma": 0.3,
        "reg_alpha": 0.3,
        "reg_lambda": 1.8,
        "random_state": 42,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
        "tree_method": "hist",
    }


if __name__ == "__main__":
    args = get_args()

    if args.min_bets_per_day <= 0:
        raise ValueError("--min_bets_per_day must be > 0")
    if args.max_bets_per_day < args.min_bets_per_day:
        raise ValueError("--max_bets_per_day must be >= --min_bets_per_day")
    if args.threshold_step <= 0:
        raise ValueError("--threshold_step must be > 0")
    if args.threshold_max < args.threshold_min:
        raise ValueError("--threshold_max must be >= --threshold_min")

    data = load_splits(args.interval)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    if args.use_weights:
        train_weights = np.clip(data["return_mag_train"] / 0.001, 0.2, 5.0)
        train_weights = train_weights / train_weights.mean()
    else:
        train_weights = None

    xgb_params = default_xgb_params()
    if args.params_json:
        with open(args.params_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "xgb_params" in payload:
            xgb_params.update(payload["xgb_params"])
        else:
            xgb_params.update(payload)

    print("\n" + "=" * 64)
    print(f"XGBoost Strategy Run — interval={args.interval}")
    print("=" * 64)
    print(f"Train/Val/Test sizes: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    print(f"Constraint: bets/day in [{args.min_bets_per_day}, {args.max_bets_per_day}], "
          f"target={args.target_bets_per_day}")
    print(f"Threshold search: [{args.threshold_min}, {args.threshold_max}] step={args.threshold_step}")
    print(f"Using sample weights: {bool(args.use_weights)}")

    model = bm.fit_xgb_with_overfit_guard(
        X_train, y_train, X_val, y_val,
        sample_weight=train_weights,
        primary_params=xgb_params,
    )

    # Feature selection: retrain on non-zero importance features only
    importances = model.feature_importances_
    keep_mask = importances > 0
    if keep_mask.sum() < len(keep_mask):
        print(f"  Feature selection: {keep_mask.sum()}/{len(keep_mask)} features kept")
        model = bm.fit_xgb_with_overfit_guard(
            X_train[:, keep_mask], y_train, X_val[:, keep_mask], y_val,
            sample_weight=train_weights,
            primary_params=xgb_params,
        )
        X_val = X_val[:, keep_mask]
        X_test = X_test[:, keep_mask]

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    thr_choice = bm.optimize_threshold_for_accuracy(
        y_true=y_val,
        y_prob=val_prob,
        num_days=data["val_days"],
        min_bpd=args.min_bets_per_day,
        max_bpd=args.max_bets_per_day,
        target_bpd=args.target_bets_per_day,
        t_min=args.threshold_min,
        t_max=args.threshold_max,
        t_step=args.threshold_step,
    )

    test_stats = bm.evaluate_threshold_on_split(
        y_true=y_test,
        y_prob=test_prob,
        threshold=thr_choice["threshold"],
        num_days=data["test_days"],
    )

    print("\nSelected threshold on val:")
    print(f"  threshold   : {thr_choice['threshold']:.3f}")
    print(f"  val_acc     : {thr_choice['accuracy']:.4f}")
    print(f"  val_bets/day: {thr_choice['bets_per_day']:.2f}")
    print(f"  val_bets    : {thr_choice['bets']}")
    print(f"  feasible    : {thr_choice['feasible']}")

    print("\nTest metrics at selected threshold:")
    print(f"  accuracy    : {test_stats['accuracy']:.4f}")
    print(f"  precision   : {test_stats['precision']:.4f}")
    print(f"  recall      : {test_stats['recall']:.4f}")
    print(f"  f1          : {test_stats['f1']:.4f}")
    print(f"  bets/day    : {test_stats['bets_per_day']:.2f}")
    print(f"  bets        : {test_stats['bets']}")
    print(f"  edge        : {test_stats['edge']:+.2f}%")

    print("\nXGBoost params used:")
    for k in sorted(xgb_params.keys()):
        print(f"  {k}: {xgb_params[k]}")

    if args.save_report:
        report = {
            "interval": args.interval,
            "xgb_params": xgb_params,
            "threshold_choice_val": thr_choice,
            "threshold_metrics_test": test_stats,
            "constraints": {
                "min_bets_per_day": args.min_bets_per_day,
                "max_bets_per_day": args.max_bets_per_day,
                "target_bets_per_day": args.target_bets_per_day,
                "threshold_min": args.threshold_min,
                "threshold_max": args.threshold_max,
                "threshold_step": args.threshold_step,
            },
        }
        save_dir = os.path.dirname(args.save_report)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to: {args.save_report}")
