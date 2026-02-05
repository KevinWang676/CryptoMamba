"""
Predict the next candle's up/down direction with probability.

Usage:
    # Using a classification model (recommended):
    python scripts/predict_updown.py \
        --config cmamba_cls_15m \
        --ckpt_path checkpoints/cmamba_cls_15m.ckpt \
        --data_path data/btc_15m.csv

    # Using a regression model (fallback):
    python scripts/predict_updown.py \
        --config cmamba_v \
        --ckpt_path checkpoints/cmamba_v.ckpt \
        --data_path data/btc_15m.csv \
        --regression
"""

import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from utils import io_tools
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT = io_tools.get_root(__file__, num_returns=2)


def get_args():
    parser = ArgumentParser(description="Predict next candle up/down with probability")
    parser.add_argument("--config", type=str, required=True,
                        help="Training config name (e.g., cmamba_cls_15m)")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to CSV data file (overrides config)")
    parser.add_argument("--regression", action="store_true", default=False,
                        help="Use regression model (predict price, then derive direction)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Minimum probability to recommend a bet (default: 0.55)")
    return parser.parse_args()


def load_model(config, ckpt_path):
    """Load model from checkpoint."""
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get('normalize', False)
    model_class = io_tools.get_obj_from_str(model_config.get('target'))
    model = model_class.load_from_checkpoint(ckpt_path, **model_config.get('params'))
    model.cuda()
    model.eval()
    return model, normalize, model_config


def prepare_features(data, model, use_volume, normalize, data_module):
    """Prepare feature tensor from the latest data."""
    features = {}
    key_list = ['Timestamp', 'Open', 'High', 'Low', 'Close']
    if use_volume:
        key_list.append('Volume')

    for key in key_list:
        tmp = list(data.get(key))
        if normalize and data_module.factors is not None:
            scale = data_module.factors.get(key).get('max') - data_module.factors.get(key).get('min')
            shift = data_module.factors.get(key).get('min')
        else:
            scale = 1
            shift = 0
        if key == 'Volume':
            tmp = [x / 1e9 for x in tmp]
        tmp = [(x - shift) / scale for x in tmp]
        features[key] = torch.tensor(tmp, dtype=torch.float32).reshape(1, -1)

    x = torch.cat([features.get(k) for k in features.keys()], dim=0)
    return x, features


def predict_classification(model, x):
    """Classification model: output logit -> sigmoid -> probability."""
    with torch.no_grad():
        logit = model.model(x[None, ...].cuda()).cpu()
        logit = float(logit.reshape(-1))
        prob_up = float(torch.sigmoid(torch.tensor(logit)))
        prob_down = 1.0 - prob_up
    return prob_up, prob_down


def predict_regression(model, x, current_close, use_volume, normalize, scale_pred, shift_pred):
    """Regression model: predict price -> derive direction."""
    with torch.no_grad():
        pred_price = float(model.model(x[None, ...].cuda()).cpu().reshape(-1))
        pred_price = pred_price * scale_pred + shift_pred

    # Derive direction from price comparison
    if pred_price > current_close:
        direction = "UP"
        confidence = min((pred_price - current_close) / current_close * 100, 5.0)
        prob_up = 0.5 + confidence / 10.0  # Simple linear scaling
        prob_up = min(prob_up, 0.95)
    else:
        direction = "DOWN"
        confidence = min((current_close - pred_price) / current_close * 100, 5.0)
        prob_up = 0.5 - confidence / 10.0
        prob_up = max(prob_up, 0.05)

    prob_down = 1.0 - prob_up
    return prob_up, prob_down, pred_price


if __name__ == "__main__":
    args = get_args()

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')
    data_config = io_tools.load_config_from_yaml(
        f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml"
    )

    use_volume = config.get('use_volume', False)
    model, normalize, model_config = load_model(config, args.ckpt_path)
    window_size = model.window_size

    # Determine the time frame from data config
    jumps = data_config.get('jumps', 86400)
    if jumps < 3600:
        tf_label = f"{jumps // 60}m"
    else:
        tf_label = f"{jumps // 3600}h"

    # Load data
    data_path = args.data_path or data_config.get('data_path')
    data = pd.read_csv(data_path)
    if 'Date' in data.keys():
        date_format = data_config.get('date_format', '%Y-%m-%d')
        data['Timestamp'] = [
            float(time.mktime(datetime.strptime(x, date_format).timetuple()))
            for x in data['Date']
        ]
    data = data.sort_values(by='Timestamp').reset_index(drop=True)

    # Take the last window_size rows as features
    if len(data) < window_size:
        print(f"Error: need at least {window_size} data points, got {len(data)}")
        sys.exit(1)

    recent_data = data.tail(window_size).reset_index(drop=True)
    current_close = float(recent_data['Close'].iloc[-1])
    current_ts = int(recent_data['Timestamp'].iloc[-1])
    current_time = datetime.fromtimestamp(current_ts)

    # Setup data module for normalization factors only if needed
    data_module = None
    if normalize:
        train_transform = DataTransform(is_train=True, use_volume=use_volume,
                                         additional_features=config.get('additional_features', []))
        data_module = CMambaDataModule(data_config,
                                        train_transform=train_transform,
                                        val_transform=train_transform,
                                        test_transform=train_transform,
                                        batch_size=1,
                                        distributed_sampler=False,
                                        num_workers=0,
                                        normalize=normalize,
                                        window_size=window_size)

    # Prepare features
    x, features = prepare_features(recent_data, model, use_volume, normalize, data_module)

    # Run prediction
    is_classification = model_config.get('params', {}).get('task', 'regression') == 'classification'
    if args.regression:
        is_classification = False

    if is_classification:
        prob_up, prob_down = predict_classification(model, x)
        pred_price = None
    else:
        # Get scale/shift for denormalization
        if normalize and data_module.factors is not None:
            scale_pred = data_module.factors.get('Close').get('max') - data_module.factors.get('Close').get('min')
            shift_pred = data_module.factors.get('Close').get('min')
        else:
            scale_pred = 1
            shift_pred = 0
        prob_up, prob_down, pred_price = predict_regression(
            model, x, current_close, use_volume, normalize, scale_pred, shift_pred
        )

    # Output results
    direction = "UP" if prob_up > 0.5 else "DOWN"

    print("")
    print("=" * 55)
    print(f"  BTC {tf_label} Candle Prediction")
    print("=" * 55)
    print(f"  Current Time:  {current_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Current Price: ${current_close:,.2f}")
    if pred_price is not None:
        print(f"  Predicted Price: ${pred_price:,.2f}")
    print(f"  Model Type:    {'Classification' if is_classification else 'Regression'}")
    print("-" * 55)
    print(f"  Prediction:    {direction}")
    print(f"  UP   Probability: {prob_up * 100:6.2f}%")
    print(f"  DOWN Probability: {prob_down * 100:6.2f}%")
    print("-" * 55)

    # Polymarket betting recommendation
    print("  Polymarket Recommendation:")
    if prob_up > args.threshold:
        edge = (prob_up - 0.5) * 100
        print(f"    -> BET UP  (edge: +{edge:.1f}%)")
        print(f"       Buy UP  if Polymarket price < {prob_up:.3f}")
    elif prob_down > args.threshold:
        edge = (prob_down - 0.5) * 100
        print(f"    -> BET DOWN (edge: +{edge:.1f}%)")
        print(f"       Buy DOWN if Polymarket price < {prob_down:.3f}")
    else:
        print(f"    -> NO BET (confidence {max(prob_up, prob_down)*100:.1f}% < threshold {args.threshold*100:.0f}%)")
    print("=" * 55)
    print("")

    # Save prediction to file
    pred_dir = f'{ROOT}/Predictions/{args.config}'
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = f'{pred_dir}/{current_time.strftime("%Y-%m-%d_%H%M")}.txt'
    with open(pred_file, 'w') as f:
        f.write(f"time: {current_time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"price: {current_close}\n")
        f.write(f"direction: {direction}\n")
        f.write(f"prob_up: {prob_up:.6f}\n")
        f.write(f"prob_down: {prob_down:.6f}\n")
        if pred_price is not None:
            f.write(f"predicted_price: {pred_price:.2f}\n")
    print(f"  Prediction saved to: {pred_file}")
