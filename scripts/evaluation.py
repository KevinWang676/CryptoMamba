import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import yaml
import torch
import matplotlib
import numpy as np
from utils import io_tools
from datetime import datetime
import pytorch_lightning as pl
import matplotlib.ticker as ticker
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
sns.set_theme(style='whitegrid', context='paper', font_scale=3)
palette = sns.color_palette('muted')



ROOT = io_tools.get_root(__file__, num_returns=2)

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        help="Logging directory.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default='gpu',
        help="The type of accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of computing devices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Logging directory.",
    )
    parser.add_argument(
        "--expname",
        type=str,
        default='Cmamba',
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='cmamba_nv',
        help="Path to config file.",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        '--use_volume',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch_size",
    )

    args = parser.parse_args()
    return args

def print_and_write(file, txt, add_new_line=True):
    print(txt)
    if add_new_line:
        file.write(f'{txt}\n')
    else:
        file.write(txt)

def save_all_hparams(log_dir, args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_dict = vars(args)
    save_dict.pop('checkpoint_callback')
    with open(log_dir + '/hparams.yaml', 'w') as f:
        yaml.dump(save_dict, f)

def init_dirs(args, name):
    path = f'{ROOT}/Results/{name}/{args.config}'
    if not os.path.isdir(path):
        os.makedirs(path)
    txt_file = open(f'{path}/metrics.txt', 'w')
    plot_path = f'{path}/pred.jpg'
    return txt_file, plot_path

def load_model(config, ckpt_path):
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

@torch.no_grad()
def run_model(model, dataloader, factors=None):
    target_list = []
    preds_list = []
    timestamps = []
    with torch.no_grad():
        for batch in dataloader:
            ts = batch.get('Timestamp').numpy().reshape(-1)
            target = batch.get(model.y_key).numpy().reshape(-1)
            features = batch.get('features').to(model.device)
            preds = model(features).cpu().numpy().reshape(-1)
            target_list += [float(x) for x in list(target)]
            preds_list += [float(x) for x in list(preds)]
            timestamps += [float(x) for x in list(ts)]

    if factors is not None:
        scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
        shift = factors.get(model.y_key).get('min')
        target_list = [x * scale + shift for x in target_list]
        preds_list = [x * scale + shift for x in preds_list]
        scale = factors.get('Timestamp').get('max') - factors.get('Timestamp').get('min')
        shift = factors.get('Timestamp').get('min')
        timestamps = [x * scale + shift for x in timestamps]
    targets = np.asarray(target_list)
    preds = np.asarray(preds_list)
    targets_tensor = torch.tensor(target_list)
    preds_tensor = torch.tensor(preds_list)
    timestamps = [datetime.fromtimestamp(int(x)) for x in timestamps]
    mse = float(model.mse(preds_tensor, targets_tensor))
    mape = float(model.mape(preds_tensor, targets_tensor))
    l1 = float(model.l1(preds_tensor, targets_tensor))
    return timestamps, targets, preds, mse, mape, l1


@torch.no_grad()
def run_model_classification(model, dataloader, factors=None):
    """Run classification model: compute accuracy, precision, recall, F1."""
    all_targets = []
    all_probs = []
    all_preds = []
    timestamps = []

    with torch.no_grad():
        for batch in dataloader:
            ts = batch.get('Timestamp').numpy().reshape(-1)
            y = batch.get(model.y_key)
            y_old = batch.get(f'{model.y_key}_old')
            features = batch.get('features').to(model.device)

            # Binary target: 1 if price went up, 0 if down
            target = (y > y_old).float()

            # Model outputs raw logits
            logits = model.model(features).reshape(-1).cpu()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_targets += target.numpy().tolist()
            all_probs += probs.numpy().tolist()
            all_preds += preds.numpy().tolist()

            # Denormalize timestamps if needed
            ts_list = ts.tolist()
            if factors is not None:
                scale = factors.get('Timestamp').get('max') - factors.get('Timestamp').get('min')
                shift = factors.get('Timestamp').get('min')
                ts_list = [x * scale + shift for x in ts_list]
            timestamps += ts_list

    targets = np.asarray(all_targets)
    probs = np.asarray(all_probs)
    preds = np.asarray(all_preds)
    timestamps = [datetime.fromtimestamp(int(x)) for x in timestamps]

    # Metrics
    n = len(targets)
    correct = (preds == targets).sum()
    accuracy = correct / n if n > 0 else 0.0

    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    up_ratio = targets.mean()

    # BCE loss
    targets_tensor = torch.tensor(all_targets)
    logits_tensor = torch.logit(torch.tensor(all_probs).clamp(1e-7, 1 - 1e-7))
    bce = float(torch.nn.functional.binary_cross_entropy_with_logits(logits_tensor, targets_tensor))

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'bce_loss': bce,
        'up_ratio': float(up_ratio),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'n': n,
    }
    return timestamps, targets, probs, preds, metrics



if __name__ == "__main__":

    args = get_args()
    pl.seed_everything(args.seed)
    logdir = args.logdir

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')
    name = config.get('name', args.expname)

    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")

    use_volume = args.use_volume
    if not use_volume:
        use_volume = config.get('use_volume')
    train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []))
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))

    model, normalize, model_config = load_model(config, args.ckpt_path)
    is_classification = model_config.get('params', {}).get('task', 'regression') == 'classification'

    data_module = CMambaDataModule(data_config,
                                   train_transform=train_transform,
                                   val_transform=val_transform,
                                   test_transform=test_transform,
                                   batch_size=args.batch_size,
                                   distributed_sampler=False,
                                   num_workers=args.num_workers,
                                   normalize=normalize,
                                   window_size=model.window_size,
                                   )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    dataloader_list = [train_loader, val_loader, test_loader]
    titles = ['Train', 'Val', 'Test']
    colors = ['red', 'green', 'magenta']

    factors = None
    if normalize:
        factors = data_module.factors

    f, plot_path = init_dirs(args, name)

    if is_classification:
        # ── Classification evaluation ──
        print_format = '{:^7} {:^10} {:^10} {:^10} {:^7} {:^10} {:^5}'
        txt = print_format.format('Split', 'Accuracy', 'Precision', 'Recall', 'F1', 'BCE', 'N')
        print_and_write(f, txt)

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        for idx, (key, dataloader, c) in enumerate(zip(titles, dataloader_list, colors)):
            timestamps, targets, probs, preds, metrics = run_model_classification(
                model, dataloader, factors
            )

            txt = print_format.format(
                key,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['bce_loss']:.4f}",
                metrics['n'],
            )
            print_and_write(f, txt)

            # Print confusion matrix
            cm_txt = (f"  {key} Confusion: TP={metrics['tp']} FP={metrics['fp']} "
                      f"FN={metrics['fn']} TN={metrics['tn']} "
                      f"UP_ratio={metrics['up_ratio']:.3f}")
            print_and_write(f, cm_txt)

            # Plot: probability distribution for correct vs incorrect predictions
            ax = axes[idx]
            correct_mask = (preds == targets)
            if correct_mask.sum() > 0:
                ax.hist(probs[correct_mask], bins=30, alpha=0.6, color='green',
                        label=f'Correct ({correct_mask.sum()})', density=True)
            if (~correct_mask).sum() > 0:
                ax.hist(probs[~correct_mask], bins=30, alpha=0.6, color='red',
                        label=f'Wrong ({(~correct_mask).sum()})', density=True)
            ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
            ax.set_title(f'{key} (Acc={metrics["accuracy"]:.3f})')
            ax.set_xlabel('P(UP)')
            ax.set_ylabel('Density')
            ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")

    else:
        # ── Regression evaluation (original logic) ──
        all_targets = []
        all_timestamps = []

        plt.figure(figsize=(20, 10))
        print_format = '{:^7} {:^15} {:^10} {:^7} {:^10}'
        txt = print_format.format('Split', 'MSE', 'RMSE', 'MAPE', 'MAE')
        print_and_write(f, txt)
        for key, dataloader, c in zip(titles, dataloader_list, colors):
            timstamps, targets, preds, mse, mape, l1 = run_model(model, dataloader, factors)
            all_timestamps += timstamps
            all_targets += list(targets)
            txt = print_format.format(key, round(mse, 3), round(np.sqrt(mse), 3), round(mape, 5), round(l1, 3))
            print_and_write(f, txt)
            sns.lineplot(x=timstamps, y=preds, color=c, linewidth=2.5, label=key)

        sns.lineplot(x=all_timestamps, y=all_targets, color='blue', zorder=0, linewidth=2.5, label='Target')
        plt.legend()
        plt.ylabel('Price ($)')
        plt.xlim([all_timestamps[0], all_timestamps[-1]])
        plt.xticks(rotation=30)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}K'.format(x/1000)))
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    f.close()
