# CryptoMamba 短周期 K 线涨跌预测指南

> 本文档详细讲解如何将 CryptoMamba 从"每日价格回归预测"改造为"短周期（15分钟/30分钟/1小时）K线涨跌分类预测"，涵盖数据获取、数据集准备、配置修改、训练与推理的完整流程。

---

## 目录

1. [架构概览](#1-架构概览)
2. [数据获取：API 选择与对比](#2-数据获取api-选择与对比)
3. [数据集准备](#3-数据集准备)
4. [配置文件修改](#4-配置文件修改)
5. [训练流程](#5-训练流程)
6. [推理与实时预测](#6-推理与实时预测)
7. [自定义 Time Frame 支持](#7-自定义-time-frame-支持)
8. [常见问题](#8-常见问题)

---

## 1. 架构概览

### 原始 CryptoMamba 数据流

```
原始 CSV (日线 OHLCV)
  → DataConverter (按 jumps=86400 秒聚合)
  → train/val/test 分割 (按时间区间)
  → CMambaDataset (滑动窗口 window_size=14)
  → DataTransform (提取特征张量)
  → CMamba 模型 (预测下一天的 Close 价格)
  → RMSE Loss (回归损失)
```

### 改造后的数据流（15分钟涨跌预测）

```
Binance API (15分钟 K 线 OHLCV)
  → 保存为 CSV
  → DataConverter (jumps=900 秒，即15分钟)
  → train/val/test 分割 (按时间区间)
  → CMambaDataset (滑动窗口 window_size=96，即24小时历史)
  → DataTransform (提取特征 + 生成涨跌标签)
  → CMamba 模型 (预测下一个 K 线涨/跌)
  → CrossEntropy Loss (分类损失)
```

### 关键修改点

| 组件 | 原始 | 改造后 |
|------|------|--------|
| 数据粒度 | 日线 (86400s) | 15分钟 (900s) |
| 窗口大小 | 14 天 | 96 个 K 线 (24小时) |
| 预测目标 | Close 价格（回归） | 涨/跌（二分类） |
| 损失函数 | RMSE | CrossEntropyLoss |
| 输出维度 | 1 | 2 (下跌概率, 上涨概率) |

---

## 2. 数据获取：API 选择与对比

### 2.1 推荐方案：Binance API（首选）

Binance 提供完全免费的 K 线数据接口，无需 API Key 即可获取历史数据。

**优势：**
- 免费，无速率限制（合理使用）
- 支持 1m / 3m / 5m / **15m** / 30m / 1h / 4h / 1d 等所有粒度
- BTC/USDT 数据从 2017 年开始
- 每次最多返回 1000 条 K 线

**获取 15 分钟 K 线数据：**

```python
import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_binance_klines(symbol="BTCUSDT", interval="15m",
                         start_date="2023-01-01", end_date="2025-12-31",
                         save_path="data/btc_15m.csv"):
    """
    从 Binance 获取历史 K 线数据。

    参数:
        symbol: 交易对，如 "BTCUSDT"
        interval: K 线周期，支持 "1m","3m","5m","15m","30m","1h","4h","1d"
        start_date: 开始日期 "YYYY-MM-DD"
        end_date: 结束日期 "YYYY-MM-DD"
        save_path: 保存路径
    """
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
            "limit": 1000  # 每次最多 1000 条
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)

        # 更新起始时间为最后一条记录的结束时间
        current_ts = data[-1][6] + 1  # close_time + 1ms

        time.sleep(0.1)  # 避免触发速率限制
        print(f"已获取 {len(all_data)} 条数据，当前时间: "
              f"{datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d %H:%M')}")

    # 转换为 DataFrame
    df = pd.DataFrame(all_data, columns=[
        "Timestamp", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_volume", "Trades",
        "Taker_buy_base", "Taker_buy_quote", "Ignore"
    ])

    # 清理数据
    df["Timestamp"] = df["Timestamp"] // 1000  # 毫秒 → 秒
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    # 只保留需要的列
    df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df = df.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    df.to_csv(save_path, index=False)
    print(f"共保存 {len(df)} 条 K 线数据到 {save_path}")
    return df


# 使用示例
if __name__ == "__main__":
    # 获取 2 年的 15 分钟 K 线数据
    df = fetch_binance_klines(
        symbol="BTCUSDT",
        interval="15m",
        start_date="2023-06-01",
        end_date="2026-02-05",
        save_path="data/btc_15m.csv"
    )
    print(df.head())
    print(f"数据范围: {datetime.fromtimestamp(df['Timestamp'].iloc[0])} "
          f"~ {datetime.fromtimestamp(df['Timestamp'].iloc[-1])}")
```

### 2.2 备选方案对比

| API | 端点示例 | 免费额度 | 15分钟支持 | 备注 |
|-----|---------|---------|-----------|------|
| **Binance** | `api.binance.com/api/v3/klines` | 无限 | ✅ | 首选，最稳定 |
| **Kraken** | 提供 CSV 批量下载 | 无限 | ✅ | 适合批量历史数据 |
| **CryptoCompare** | `min-api.cryptocompare.com/data/v2/histominute` | 有限 | ✅ | 需聚合1分钟数据 |
| **CoinGecko** | `api.coingecko.com/api/v3/coins/{id}/ohlc` | 有限 | 付费 | 免费版粒度有限 |
| **CoinAPI** | `rest.coinapi.io/v1/ohlcv/{symbol}/history` | 100次/天 | ✅ | 多交易所聚合 |

> **注意**: Yahoo Finance **不推荐**用于此任务。它的加密货币数据最小粒度为日线，且API不稳定。

### 2.3 数据格式要求

CryptoMamba 需要的 CSV 格式：

```csv
Timestamp,Open,High,Low,Close,Volume
1706745600,42000.50,42150.00,41980.00,42100.00,1500.5
1706746500,42100.00,42200.00,42050.00,42180.00,1200.3
...
```

**字段说明：**
- `Timestamp`: Unix 时间戳（秒）
- `Open/High/Low/Close`: OHLC 价格（浮点数）
- `Volume`: 成交量（浮点数）

如果你的原始数据带有 `Date` 列而非 `Timestamp`，需要使用包含时分秒的日期格式。

---

## 3. 数据集准备

### 3.1 数据量建议

| Time Frame | 每天数据量 | 推荐最少训练集 | 对应天数 |
|------------|-----------|--------------|---------|
| 15 分钟 | 96 条 | ~50,000 条 | ~520 天 |
| 30 分钟 | 48 条 | ~30,000 条 | ~625 天 |
| 1 小时 | 24 条 | ~20,000 条 | ~833 天 |

> 加密货币市场 7×24 不间断交易，所以每天都是完整的 96 条 15 分钟 K 线。

### 3.2 创建数据配置文件

为 15 分钟 K 线创建新的数据配置：

**新建文件 `configs/data_configs/btc_15m.yaml`：**

```yaml
# 15 分钟 K 线配置
jumps: 900                           # 15分钟 = 900秒
start_date: "2023-06-01 00:00:00"    # 数据起始时间
end_date: "2026-02-05 00:00:00"      # 数据结束时间
root: "./data"
date_format: "%Y-%m-%d %H:%M:%S"    # 注意：需要精确到秒
data_path: "./data/btc_15m.csv"      # 你的原始数据文件路径

# 时间序列分割（按时间顺序，不能随机打乱）
train_interval: ["2023-06-01 00:00:00", "2025-06-01 00:00:00"]   # ~2年训练集
val_interval:   ["2025-06-01 00:00:00", "2025-10-01 00:00:00"]   # ~4个月验证集
test_interval:  ["2025-10-01 00:00:00", "2026-02-05 00:00:00"]   # ~4个月测试集

additional_features: []
```

### 3.3 其他 Time Frame 的配置示例

**30 分钟 — `configs/data_configs/btc_30m.yaml`：**

```yaml
jumps: 1800
start_date: "2023-06-01 00:00:00"
end_date: "2026-02-05 00:00:00"
root: "./data"
date_format: "%Y-%m-%d %H:%M:%S"
data_path: "./data/btc_30m.csv"
train_interval: ["2023-06-01 00:00:00", "2025-06-01 00:00:00"]
val_interval:   ["2025-06-01 00:00:00", "2025-10-01 00:00:00"]
test_interval:  ["2025-10-01 00:00:00", "2026-02-05 00:00:00"]
additional_features: []
```

**1 小时 — `configs/data_configs/btc_1h.yaml`：**

```yaml
jumps: 3600
start_date: "2023-06-01 00:00:00"
end_date: "2026-02-05 00:00:00"
root: "./data"
date_format: "%Y-%m-%d %H:%M:%S"
data_path: "./data/btc_1h.csv"
train_interval: ["2023-06-01 00:00:00", "2025-06-01 00:00:00"]
val_interval:   ["2025-06-01 00:00:00", "2025-10-01 00:00:00"]
test_interval:  ["2025-10-01 00:00:00", "2026-02-05 00:00:00"]
additional_features: []
```

### 3.4 关于 `jumps` 和原始数据粒度的关系

`DataConverter.process_data()` 的工作方式是：从你的**原始数据**中，按 `jumps` 秒为间隔，聚合生成新的 OHLCV 数据。

- 如果你的原始数据已经是 15 分钟粒度，设 `jumps=900` 即可直接使用
- 如果你的原始数据是 1 分钟粒度，设 `jumps=900` 会自动聚合每 15 根 1 分钟 K 线为 1 根 15 分钟 K 线
- **原始数据粒度必须 ≤ jumps**

---

## 4. 配置文件修改

### 4.1 模型配置

对于 15 分钟预测，你需要调整 `window_size` 和模型维度。

关键规则：**`window_size` 必须等于 `hidden_dims[0]`**（代码中有 `assert window_size == hidden_dims[0]`）。

**新建文件 `configs/models/CryptoMamba/v2_15m.yaml`：**

```yaml
target: pl_modules.cmamba_module.CryptoMambaModule
params:
    num_features: 6            # Timestamp, Open, High, Low, Close, Volume
    hidden_dims: [96, 64, 32, 1]   # 96 = window_size (24小时 × 4 = 96个15分钟K线)
    d_states: 64               # SSM 隐藏状态维度
    layer_density: 4           # 每层 Mamba block 数量
    loss: 'rmse'               # 如果做回归预测
    # loss: 'mse'              # 也可使用 MSE
normalize: False
```

**窗口大小建议：**

| Time Frame | 推荐 window_size | 含义 | hidden_dims[0] |
|-----------|-----------------|------|----------------|
| 15 分钟 | 96 | 看过去 24 小时 | 96 |
| 30 分钟 | 48 | 看过去 24 小时 | 48 |
| 1 小时 | 48 | 看过去 48 小时 | 48 |

> 你也可以尝试其他窗口大小（如 192 = 48 小时），但更大的窗口需要更多显存和训练时间。

### 4.2 在 archs.yaml 中注册新模型

**编辑 `configs/models/archs.yaml`，添加：**

```yaml
CMamba_v2_15m: 'CryptoMamba/v2_15m.yaml'
```

### 4.3 训练配置

**新建文件 `configs/training/cmamba_15m.yaml`：**

```yaml
data_config: 'btc_15m'          # 指向你的数据配置
model: 'CMamba_v2_15m'          # 指向你的模型配置

name: 'CMamba_15m'
max_epochs: 500                 # 数据量大，不需要太多 epoch
use_volume: True
additional_features: []

hyperparams:
    optimizer: 'adam'
    lr: 0.001                   # 比原始更小的学习率（数据量更大）
    lr_step_size: 100
    lr_gamma: 0.5
    weight_decay: 0.001
```

---

## 5. 训练流程

### 5.1 完整步骤

```bash
# Step 1: 获取数据（运行你的数据下载脚本）
python scripts/fetch_data.py  # 你自己编写的数据获取脚本

# Step 2: 验证数据格式
head -5 data/btc_15m.csv
# 应该看到:
# Timestamp,Open,High,Low,Close,Volume
# 1685577600,27150.5,27200.0,27100.0,27180.3,1500.5

# Step 3: 训练模型
python3 scripts/training.py \
    --config cmamba_15m \
    --save_checkpoints \
    --use_volume \
    --max_epochs 500 \
    --batch_size 64 \
    --accelerator gpu \
    --devices 1 \
    --logger_type tb

# Step 4: 监控训练（可选）
tensorboard --logdir logs/

# Step 5: 评估模型
python scripts/evaluation.py \
    --config cmamba_15m \
    --ckpt_path logs/CMamba_15m/version_0/checkpoints/best.ckpt

# Step 6: 交易模拟
python scripts/simulate_trade.py \
    --config cmamba_15m \
    --ckpt_path logs/CMamba_15m/version_0/checkpoints/best.ckpt \
    --split test \
    --trade_mode smart
```

### 5.2 训练参数调优建议

| 参数 | 日线原始值 | 15分钟建议值 | 原因 |
|------|----------|-------------|------|
| `batch_size` | 32 | 64-128 | 数据量大，可增加 batch |
| `lr` | 0.01 | 0.001 | 数据密度高，需更小学习率 |
| `max_epochs` | 1000 | 300-500 | 数据量大，更快收敛 |
| `lr_step_size` | 100 | 50-100 | 配合 epoch 数调整 |
| `window_size` | 14 | 96 | 24小时历史 |
| `weight_decay` | 0.001 | 0.001 | 保持不变 |

### 5.3 训练过程中的关键指标

```
Epoch 100: train/rmse=150.2, val/rmse=165.3, train/mape=0.15%, val/mape=0.17%
Epoch 200: train/rmse=120.1, val/rmse=155.8, train/mape=0.12%, val/mape=0.16%
Epoch 300: train/rmse=95.5,  val/rmse=152.1, train/mape=0.09%, val/mape=0.15%
```

- 如果 `val/rmse` 持续不下降而 `train/rmse` 在下降 → 过拟合，考虑增加 `weight_decay` 或减小模型
- 如果两者都不下降 → 学习率太小或模型容量不够

---

## 6. 推理与实时预测

### 6.1 单次预测（当前模型方式）

当前 `one_day_pred.py` 的逻辑是：
1. 读取最近 `window_size` 条 K 线数据
2. 输入模型，预测下一条 K 线的 Close 价格
3. 对比当前价格，给出涨跌信号

对于 15 分钟预测，你需要：

```bash
# 预测下一个 15 分钟 K 线
python scripts/one_day_pred.py \
    --config cmamba_15m \
    --ckpt_path checkpoints/cmamba_15m.ckpt \
    --data_path data/btc_15m_latest.csv
```

**注意**：`one_day_pred.py` 中有一行硬编码了 `14 * 24 * 60 * 60`（14天），你需要根据 window_size 修改这个值。对于 window_size=96 的 15 分钟预测，应改为 `96 * 15 * 60`（即 96 个 15 分钟的秒数）。

### 6.2 涨跌判断逻辑

当前模型预测的是**下一条 K 线的 Close 价格**（回归任务），涨跌判断方式：

```python
predicted_close = model(features)   # 模型输出
current_close = latest_kline.close  # 当前 K 线收盘价

if predicted_close > current_close:
    signal = "UP"    # 预测上涨
else:
    signal = "DOWN"  # 预测下跌

# 涨跌幅度
change_pct = (predicted_close - current_close) / current_close * 100
```

### 6.3 定时推理（每 15 分钟自动预测）

如果你想每 15 分钟自动运行一次预测：

```python
import schedule
import time
import subprocess

def run_prediction():
    """每 15 分钟运行一次"""
    # 1. 先更新最新数据
    # （调用你的数据获取脚本获取最近的 K 线）

    # 2. 运行预测
    subprocess.run([
        "python", "scripts/one_day_pred.py",
        "--config", "cmamba_15m",
        "--ckpt_path", "checkpoints/cmamba_15m.ckpt",
        "--data_path", "data/btc_15m_latest.csv"
    ])

# 在每个 15 分钟整点运行
for minute in ["00", "15", "30", "45"]:
    schedule.every().hour.at(f":{minute}").do(run_prediction)

while True:
    schedule.run_pending()
    time.sleep(1)
```

---

## 7. 自定义 Time Frame 支持

### 7.1 快速配置表

只需修改 3 个文件即可支持任意 time frame：

#### 15 分钟

| 文件 | 关键参数 |
|------|---------|
| `configs/data_configs/btc_15m.yaml` | `jumps: 900`, `date_format: "%Y-%m-%d %H:%M:%S"` |
| `configs/models/CryptoMamba/v2_15m.yaml` | `hidden_dims: [96, 64, 32, 1]`, `num_features: 6` |
| `configs/training/cmamba_15m.yaml` | `data_config: 'btc_15m'`, `model: 'CMamba_v2_15m'` |

#### 30 分钟

| 文件 | 关键参数 |
|------|---------|
| `configs/data_configs/btc_30m.yaml` | `jumps: 1800` |
| `configs/models/CryptoMamba/v2_30m.yaml` | `hidden_dims: [48, 64, 32, 1]` |
| `configs/training/cmamba_30m.yaml` | `data_config: 'btc_30m'`, `model: 'CMamba_v2_30m'` |

#### 45 分钟

| 文件 | 关键参数 |
|------|---------|
| `configs/data_configs/btc_45m.yaml` | `jumps: 2700` |
| `configs/models/CryptoMamba/v2_45m.yaml` | `hidden_dims: [32, 64, 32, 1]` |
| `configs/training/cmamba_45m.yaml` | `data_config: 'btc_45m'`, `model: 'CMamba_v2_45m'` |

> 注意：Binance API 不直接支持 45 分钟 K 线。你需要下载 15 分钟数据，
> 然后依赖 DataConverter 的 `jumps: 2700` 自动聚合为 45 分钟。

#### 1 小时

| 文件 | 关键参数 |
|------|---------|
| `configs/data_configs/btc_1h.yaml` | `jumps: 3600` |
| `configs/models/CryptoMamba/v2_1h.yaml` | `hidden_dims: [48, 64, 32, 1]` |
| `configs/training/cmamba_1h.yaml` | `data_config: 'btc_1h'`, `model: 'CMamba_v2_1h'` |

### 7.2 通用公式

```
window_size 的选择:
  目标 = 覆盖过去 N 小时的历史数据
  window_size = N × (3600 / jumps)

举例:
  15分钟K线，看过去24小时: window_size = 24 × (3600/900)  = 96
  30分钟K线，看过去24小时: window_size = 24 × (3600/1800) = 48
  1小时K线，看过去48小时:  window_size = 48 × (3600/3600) = 48

hidden_dims[0] = window_size（这是硬约束）
hidden_dims[-1] = 1（回归输出）
中间层自由设计，建议 2-3 层，逐步缩小或先扩大再缩小
```

---

## 8. 常见问题

### Q1: `date_format` 报错

如果你的数据 Timestamp 是 Unix 秒而不是日期字符串，确保 CSV 中有 `Timestamp` 列（整数），DataConverter 会自动识别。此时 `date_format` 仍用于 `start_date` / `end_date` 的解析。

### Q2: `assert window_size == hidden_dims[0]` 失败

这是 `CryptoMambaModule.__init__()` 中的约束。确保你的模型配置中 `hidden_dims` 的第一个元素等于数据配置暗示的 `window_size`。`window_size` 默认值是 `hidden_dims[0]`。

### Q3: 显存不足

15 分钟数据的 window_size=96 比原始的 14 大很多。如果显存不足：
- 减小 `batch_size`（如 32 → 16）
- 减小 `d_states`（如 64 → 32）
- 减小 `layer_density`（如 4 → 2）
- 减小中间 `hidden_dims`（如 `[96, 32, 1]`）

### Q4: 训练不收敛

- 检查数据中是否有 NaN 或 0 值
- 降低学习率（0.001 → 0.0001）
- 增加 `weight_decay`（0.001 → 0.01）
- 确保数据按时间顺序排列

### Q5: 预测精度不理想

- 增加训练数据量（至少 6 个月以上）
- 尝试添加技术指标作为 `additional_features`
- 尝试 `normalize: True` 启用归一化
- 实验不同的 `window_size`（更长的历史可能更好）
- 15 分钟预测本身噪声很大，MAPE 在 0.1-0.5% 范围内就已经不错
