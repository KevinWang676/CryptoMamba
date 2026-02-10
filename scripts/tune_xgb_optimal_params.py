"""
Search XGBoost hyperparameters + confidence threshold for BTC up/down prediction.

Selection objective on validation split:
  1) maximize thresholded accuracy
  2) keep bets/day within [min_bets_per_day, max_bets_per_day]
  3) prefer target_bets_per_day within feasible candidates

Usage:
    python scripts/tune_xgb_optimal_params.py --interval 15m --trials 120
"""

import os
import sys
import json
import pathlib

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from argparse import ArgumentParser
from sklearn.model_selection import TimeSeriesSplit

from scripts import baseline_models as bm


def get_args():
    parser = ArgumentParser(description="Tune XGBoost params with threshold-aware objective")
    parser.add_argument("--interval", type=str, default="15m", choices=["15m", "30m", "1h"])
    parser.add_argument("--trials", type=int, default=120, help="Number of random-search trials")
    parser.add_argument("--seed", type=int, default=20260210)
    parser.add_argument("--use_weights", action="store_true", default=False,
                        help="Use return-magnitude sample weights")
    parser.add_argument("--min_bets_per_day", type=float, default=8.0)
    parser.add_argument("--max_bets_per_day", type=float, default=20.0)
    parser.add_argument("--target_bets_per_day", type=float, default=12.0)
    parser.add_argument("--threshold_min", type=float, default=0.50)
    parser.add_argument("--threshold_max", type=float, default=0.65)
    parser.add_argument("--threshold_step", type=float, default=0.001)
    parser.add_argument("--save_json", type=str, default="Results/xgb_best_15m.json",
                        help="Where to save tuned params and selected threshold")
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
    tv_mask = (ts >= train_s) & (ts < val_e)

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
        "X_tv": feats.loc[feats.index[tv_mask], feature_cols].values,
        "y_tv": feats.loc[feats.index[tv_mask], "target"].values,
        "base_seconds": cfg["jumps"],
        "return_mag_train": return_mag_train,
        "val_days": val_days, "test_days": test_days,
    }


def baseline_xgb_params():
    return {
        "n_estimators": 1400,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 12,
        "gamma": 0.3,
        "reg_alpha": 0.3,
        "reg_lambda": 1.8,
        "random_state": 42,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
        "tree_method": "hist",
    }


def sample_params(rng):
    return {
        "n_estimators": int(rng.integers(1000, 3601)),
        "max_depth": int(rng.integers(3, 10)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.005), np.log(0.06)))),
        "subsample": float(rng.uniform(0.60, 0.98)),
        "colsample_bytree": float(rng.uniform(0.50, 0.98)),
        "colsample_bylevel": float(rng.uniform(0.50, 0.98)),
        "min_child_weight": float(np.exp(rng.uniform(np.log(2.0), np.log(80.0)))),
        "gamma": float(rng.uniform(0.0, 8.0)),
        "reg_alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(20.0)))),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.2), np.log(20.0)))),
        "random_state": 42,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
        "tree_method": "hist",
    }


def evaluate_candidate(params, split_data, sample_weight, args):
    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]
    X_test, y_test = split_data["X_test"], split_data["y_test"]

    model = xgb.XGBClassifier(**params)
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_train, y_train), (X_val, y_val)],
        "verbose": 0,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(**fit_kwargs)

    snap = bm._xgb_eval_snapshot(model)
    val_prob = model.predict_proba(X_val)[:, 1]
    choice = bm.optimize_threshold_for_accuracy(
        y_true=y_val,
        y_prob=val_prob,
        num_days=split_data["val_days"],
        min_bpd=args.min_bets_per_day,
        max_bpd=args.max_bets_per_day,
        target_bpd=args.target_bets_per_day,
        t_min=args.threshold_min,
        t_max=args.threshold_max,
        t_step=args.threshold_step,
    )

    # Extra overfit penalty for candidate ranking (in addition to threshold objective).
    gap = snap["gap"] if np.isfinite(snap["gap"]) else np.inf
    overfit_penalty = 0.0 if gap <= 0.010 else 0.25 * (gap - 0.010)
    rank_score = choice["objective"] - overfit_penalty

    test_prob = model.predict_proba(X_test)[:, 1]
    test_stats = bm.evaluate_threshold_on_split(
        y_true=y_test,
        y_prob=test_prob,
        threshold=choice["threshold"],
        num_days=split_data["test_days"],
    )

    return {
        "rank_score": float(rank_score),
        "objective": float(choice["objective"]),
        "choice": choice,
        "test_stats": test_stats,
        "snap": snap,
    }


def evaluate_robust_cv(params, split_data, args, n_splits=3):
    """Robustness check with rolling time-series CV on train+val period."""
    X_tv, y_tv = split_data["X_tv"], split_data["y_tv"]
    base_seconds = split_data["base_seconds"]
    cv = TimeSeriesSplit(n_splits=n_splits)

    fold_acc = []
    fold_bpd = []
    for tr_idx, va_idx in cv.split(X_tv):
        X_tr, y_tr = X_tv[tr_idx], y_tv[tr_idx]
        X_va, y_va = X_tv[va_idx], y_tv[va_idx]
        num_days = len(va_idx) * base_seconds / 86400.0

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
        va_prob = model.predict_proba(X_va)[:, 1]
        choice = bm.optimize_threshold_for_accuracy(
            y_true=y_va,
            y_prob=va_prob,
            num_days=num_days,
            min_bpd=args.min_bets_per_day,
            max_bpd=args.max_bets_per_day,
            target_bpd=args.target_bets_per_day,
            t_min=args.threshold_min,
            t_max=args.threshold_max,
            t_step=args.threshold_step,
        )
        stats = bm.evaluate_threshold_on_split(
            y_true=y_va,
            y_prob=va_prob,
            threshold=choice["threshold"],
            num_days=num_days,
        )
        fold_acc.append(stats["accuracy"])
        fold_bpd.append(stats["bets_per_day"])

    return {
        "mean_acc": float(np.nanmean(fold_acc)),
        "std_acc": float(np.nanstd(fold_acc)),
        "mean_bpd": float(np.nanmean(fold_bpd)),
    }


if __name__ == "__main__":
    args = get_args()

    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.min_bets_per_day <= 0:
        raise ValueError("--min_bets_per_day must be > 0")
    if args.max_bets_per_day < args.min_bets_per_day:
        raise ValueError("--max_bets_per_day must be >= --min_bets_per_day")
    if args.threshold_step <= 0:
        raise ValueError("--threshold_step must be > 0")
    if args.threshold_max < args.threshold_min:
        raise ValueError("--threshold_max must be >= --threshold_min")

    data = load_splits(args.interval)
    if args.use_weights:
        train_weights = np.clip(data["return_mag_train"] / 0.001, 0.2, 5.0)
        train_weights = train_weights / train_weights.mean()
    else:
        train_weights = None

    print("\n" + "=" * 68)
    print(f"XGBoost Hyperparameter Search — interval={args.interval}")
    print("=" * 68)
    print(f"Trials: {args.trials}, seed: {args.seed}")
    print(f"Constraint: bets/day in [{args.min_bets_per_day}, {args.max_bets_per_day}], "
          f"target={args.target_bets_per_day}")
    print(f"Threshold search: [{args.threshold_min}, {args.threshold_max}] step={args.threshold_step}")
    print(f"Using sample weights: {bool(args.use_weights)}")
    print(f"Train/Val/Test sizes: {len(data['X_train'])} / {len(data['X_val'])} / {len(data['X_test'])}")

    rng = np.random.default_rng(args.seed)

    baseline_params = baseline_xgb_params()
    baseline_eval = evaluate_candidate(baseline_params, data, train_weights, args)
    best = {
        "params": baseline_params,
        **baseline_eval,
        "tag": "baseline",
        "trial": 0,
    }

    print("\nBaseline:")
    print(f"  rank_score={best['rank_score']:.6f} objective={best['objective']:.6f} "
          f"val_acc={best['choice']['accuracy']:.4f} val_bpd={best['choice']['bets_per_day']:.2f} "
          f"thr={best['choice']['threshold']:.3f} gap={best['snap']['gap']:.5f}")
    print(f"  test_acc={best['test_stats']['accuracy']:.4f} "
          f"test_bpd={best['test_stats']['bets_per_day']:.2f} "
          f"bets={best['test_stats']['bets']} edge={best['test_stats']['edge']:+.2f}%")

    for i in range(1, args.trials + 1):
        params = sample_params(rng)
        trial_eval = evaluate_candidate(params, data, train_weights, args)
        if trial_eval["rank_score"] > best["rank_score"]:
            best = {"params": params, **trial_eval, "tag": "random", "trial": i}

        if i % 10 == 0:
            print(f"  trial {i:>3d}/{args.trials}: current_best_score={best['rank_score']:.6f} "
                  f"val_acc={best['choice']['accuracy']:.4f} "
                  f"val_bpd={best['choice']['bets_per_day']:.2f} "
                  f"test_acc={best['test_stats']['accuracy']:.4f}")

    print("\nBest candidate (selected on validation objective):")
    print(f"  source={best['tag']} trial={best['trial']}")
    print(f"  rank_score={best['rank_score']:.6f} objective={best['objective']:.6f}")
    print(f"  val_acc={best['choice']['accuracy']:.4f} "
          f"val_bpd={best['choice']['bets_per_day']:.2f} "
          f"val_bets={best['choice']['bets']} "
          f"thr={best['choice']['threshold']:.3f} feasible={best['choice']['feasible']}")
    print(f"  gap(train/val logloss)={best['snap']['gap']:.5f} "
          f"(train_ll={best['snap']['train_ll']:.5f}, val_ll={best['snap']['val_ll']:.5f})")
    print(f"  test_acc={best['test_stats']['accuracy']:.4f} "
          f"test_bpd={best['test_stats']['bets_per_day']:.2f} "
          f"test_bets={best['test_stats']['bets']} edge={best['test_stats']['edge']:+.2f}%")

    print("\nRobustness check (rolling CV on train+val):")
    robust_base = evaluate_robust_cv(baseline_params, data, args, n_splits=3)
    robust_cand = evaluate_robust_cv(best["params"], data, args, n_splits=3)
    print(f"  baseline: mean_acc={robust_base['mean_acc']:.4f} ± {robust_base['std_acc']:.4f}, "
          f"mean_bpd={robust_base['mean_bpd']:.2f}")
    print(f"  candidate: mean_acc={robust_cand['mean_acc']:.4f} ± {robust_cand['std_acc']:.4f}, "
          f"mean_bpd={robust_cand['mean_bpd']:.2f}")

    # Candidate replaces baseline only if robust mean accuracy is clearly better.
    robust_margin = 0.0010
    if robust_cand["mean_acc"] >= robust_base["mean_acc"] + robust_margin:
        final_tag = "candidate"
        final_params = best["params"]
        print("  final selection: candidate (passes robustness margin)")
    else:
        final_tag = "baseline"
        final_params = baseline_params
        print("  final selection: baseline (candidate rejected as unstable/overfit)")

    final_eval = evaluate_candidate(final_params, data, train_weights, args)
    print("\nFinal selected params:")
    for k in sorted(final_params.keys()):
        print(f"  {k}: {final_params[k]}")
    print(f"  final_val_acc={final_eval['choice']['accuracy']:.4f} "
          f"final_val_bpd={final_eval['choice']['bets_per_day']:.2f} "
          f"final_thr={final_eval['choice']['threshold']:.3f}")
    print(f"  final_test_acc={final_eval['test_stats']['accuracy']:.4f} "
          f"final_test_bpd={final_eval['test_stats']['bets_per_day']:.2f} "
          f"final_test_edge={final_eval['test_stats']['edge']:+.2f}%")

    payload = {
        "interval": args.interval,
        "objective_best_candidate": {
            "selected_from": {"tag": best["tag"], "trial": best["trial"]},
            "xgb_params": best["params"],
            "val_selection": best["choice"],
            "test_metrics_at_selected_threshold": best["test_stats"],
            "overfit_diagnostics": best["snap"],
        },
        "robustness_cv": {
            "baseline": robust_base,
            "candidate": robust_cand,
            "robust_margin": robust_margin,
        },
        "final_selection": {
            "tag": final_tag,
            "xgb_params": final_params,
            "val_selection": final_eval["choice"],
            "test_metrics_at_selected_threshold": final_eval["test_stats"],
            "overfit_diagnostics": final_eval["snap"],
        },
        "xgb_params": final_params,
        "selected_threshold": final_eval["choice"]["threshold"],
        "search_config": {
            "trials": args.trials,
            "seed": args.seed,
            "use_weights": bool(args.use_weights),
            "min_bets_per_day": args.min_bets_per_day,
            "max_bets_per_day": args.max_bets_per_day,
            "target_bets_per_day": args.target_bets_per_day,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
        },
    }

    save_dir = os.path.dirname(args.save_json)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved best configuration to: {args.save_json}")
