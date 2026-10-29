"""
AI Trading System — Model Training with Purged K-Fold Cross-Validation & Meta-Labeling
Trains LightGBM + ScalpNet (Huber Loss regression + Softplus non-negativity)
Supports data from QuestDB with automated fallback to MT5 copy_rates_range.
"""
import argparse
import logging
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.indicators import (
    compute_scalp_features_from_df,
    LOOKBACK_WINDOW,
    FEATURE_NAMES
)
from agents.scalp_signal import ScalpNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_model")


class PurgedEmbargoKFold:
    """
    Purged K-Fold Cross-Validation with Embargo (Marcos López de Prado).
    Prevents leakage from overlapping labeling windows (15 bars M5).
    """
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray, t0: pd.Series, t1: pd.Series):
        indices = np.arange(len(X))
        embargo_size = int(len(X) * self.embargo_pct)
        test_ranges = np.array_split(indices, self.n_splits)

        for test_idx in test_ranges:
            t0_test_start = t0.iloc[test_idx[0]]
            t1_test_max = t1.iloc[test_idx].max()

            # Purge training samples whose lifespan overlaps the test period
            train_mask = (t1 < t0_test_start) | (t0 > t1_test_max)

            # Apply Embargo immediately after the test partition
            embargo_end_idx = min(len(X) - 1, test_idx[-1] + embargo_size)
            train_mask.iloc[test_idx[0] : embargo_end_idx + 1] = False

            train_idx = indices[train_mask.to_numpy()]
            yield train_idx, test_idx


def fetch_candles_hybrid(symbol: str, days: int = 60) -> pd.DataFrame:
    """Fetches M1/M5 candles from QuestDB, falling back to MT5 directly."""
    # 1. Try QuestDB
    try:
        query = f"SELECT * FROM candles WHERE symbol = '{symbol}' AND timestamp >= dateadd('d', -{days}, now()) ORDER BY timestamp ASC"
        resp = requests.get(f"http://192.168.1.90:9000/exec", params={"query": query}, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            cols = [c['name'] for c in data.get('columns', [])]
            df = pd.DataFrame(data.get('dataset', []), columns=cols)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                logger.info(f"Loaded {len(df)} bars from QuestDB for {symbol}")
                return _add_atr_to_df(df, 14)
    except Exception as e:
        logger.debug(f"QuestDB not reachable: {e}. Falling back to MT5 API.")

    # 2. Fallback to MT5 API directly
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            logger.error("MT5 init failed during data fetch.")
            return pd.DataFrame()

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_dt, end_dt)
        mt5.shutdown()

        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s")
            df["spread_price"] = df.get("spread", 10) * 0.00001
            logger.info(f"Loaded {len(df)} M5 bars directly from MT5 terminal for {symbol}")
            return _add_atr_to_df(df, 14)
    except Exception as e:
        logger.error(f"MT5 fetch error: {e}")

    return pd.DataFrame()


def _add_atr_to_df(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    trs = np.zeros(len(df))
    for i in range(1, len(df)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        trs[i] = max(tr1, tr2, tr3)

    atr = np.zeros(len(df))
    for i in range(period, len(df)):
        atr[i] = np.mean(trs[i - period:i])
    df[f"atr_{period}"] = atr
    return df


def triple_barrier_label(
    df: pd.DataFrame,
    idx: int,
    sl_mult: float = 1.0,
    tp_mult: float = 1.25,
    max_holding: int = 15
) -> Optional[Tuple[int, float, float, float, float]]:
    if idx + max_holding >= len(df):
        return None

    atr = df.iloc[idx].get("atr_14", 0.0)
    close = df.iloc[idx]["close"]
    spread = df.iloc[idx].get("spread_price", 0.0)
    half_spread = spread / 2.0

    if atr <= 0:
        return None

    future = df.iloc[idx + 1 : idx + 1 + max_holding]

    # BUY
    buy_entry = close + half_spread
    buy_tp = buy_entry + atr * tp_mult
    buy_sl = buy_entry - atr * sl_mult

    buy_mfe, buy_mae = 0.0, 0.0
    for _, bar in future.iterrows():
        bid_high = bar["high"] - half_spread
        bid_low = bar["low"] - half_spread
        buy_mfe = max(buy_mfe, (bid_high - buy_entry) / atr)
        buy_mae = max(buy_mae, (buy_entry - bid_low) / atr)

    buy_result = "TIMEOUT"
    buy_exit = future.iloc[-1]["close"] - half_spread
    for _, bar in future.iterrows():
        bid_low = bar["low"] - half_spread
        bid_high = bar["high"] - half_spread
        if bid_low <= buy_sl:
            buy_result = "SL"
            buy_exit = buy_sl
            break
        if bid_high >= buy_tp:
            buy_result = "TP"
            buy_exit = buy_tp
            break
    buy_pnl_atr = (buy_exit - buy_entry) / atr

    # SELL
    sell_entry = close - half_spread
    sell_tp = sell_entry - atr * tp_mult
    sell_sl = sell_entry + atr * sl_mult

    sell_mfe, sell_mae = 0.0, 0.0
    for _, bar in future.iterrows():
        ask_low = bar["low"] + half_spread
        ask_high = bar["high"] + half_spread
        sell_mfe = max(sell_mfe, (sell_entry - ask_low) / atr)
        sell_mae = max(sell_mae, (ask_high - sell_entry) / atr)

    sell_result = "TIMEOUT"
    sell_exit = future.iloc[-1]["close"] + half_spread
    for _, bar in future.iterrows():
        ask_high = bar["high"] + half_spread
        ask_low = bar["low"] + half_spread
        if ask_high >= sell_sl:
            sell_result = "SL"
            sell_exit = sell_sl
            break
        if ask_low <= sell_tp:
            sell_result = "TP"
            sell_exit = sell_tp
            break
    sell_pnl_atr = (sell_entry - sell_exit) / atr

    # LABEL: 0=BUY, 1=SELL, 2=HOLD
    if buy_result == "TP" and sell_result != "TP":
        return (0, buy_mfe, buy_mae, buy_pnl_atr, sell_pnl_atr)
    elif sell_result == "TP" and buy_result != "TP":
        return (1, sell_mfe, sell_mae, buy_pnl_atr, sell_pnl_atr)
    else:
        return (2, 0.0, 0.0, buy_pnl_atr, sell_pnl_atr)


def compute_sample_uniqueness(n_samples: int, max_holding: int) -> np.ndarray:
    weights = np.ones(n_samples, dtype=np.float32)
    max_bar = n_samples + max_holding
    bar_counts = np.zeros(max_bar, dtype=np.float32)
    for i in range(n_samples):
        bar_counts[i + 1 : i + 1 + max_holding] += 1.0

    for i in range(n_samples):
        bars = bar_counts[i + 1 : i + 1 + max_holding]
        bars = bars[bars > 0]
        if len(bars) > 0:
            weights[i] = np.mean(1.0 / bars)

    if np.sum(weights) > 0:
        weights *= n_samples / weights.sum()
    return weights


def prepare_dataset(df: pd.DataFrame, max_holding: int = 15, subsample: bool = False) -> Optional[Dict[str, np.ndarray]]:
    features, labels, mfes, maes = [], [], [], []
    buy_pnls, sell_pnls = [], []
    t0_list, t1_list = [], []

    step = max_holding if subsample else 1
    valid_indices = range(LOOKBACK_WINDOW, len(df) - max_holding, step)

    for i in valid_indices:
        f = compute_scalp_features_from_df(df, i)
        if f is None:
            continue

        result = triple_barrier_label(df, i)
        if result is None:
            continue

        label, mfe, mae, buy_pnl, sell_pnl = result
        features.append(f)
        labels.append(label)
        mfes.append(mfe)
        maes.append(mae)
        buy_pnls.append(buy_pnl)
        sell_pnls.append(sell_pnl)
        t0_list.append(df.iloc[i]["timestamp"])
        t1_list.append(df.iloc[min(len(df) - 1, i + max_holding)]["timestamp"])

    if len(features) < 50:
        return None

    out = {
        "X": np.array(features, dtype=np.float32),
        "y": np.array(labels, dtype=np.int64),
        "mfe": np.array(mfes, dtype=np.float32),
        "mae": np.array(maes, dtype=np.float32),
        "buy_pnl": np.array(buy_pnls, dtype=np.float32),
        "sell_pnl": np.array(sell_pnls, dtype=np.float32),
        "t0": pd.Series(t0_list),
        "t1": pd.Series(t1_list)
    }

    if not subsample:
        out["sample_weight"] = compute_sample_uniqueness(len(features), max_holding)

    return out


def train_scalpnet_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mfe_train: np.ndarray,
    mae_train: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    n_features: int = 34
) -> ScalpNet:
    """Trains ScalpNet with Huber Loss (SmoothL1Loss) and Softplus excursion bounds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScalpNet(n_features=n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.SmoothL1Loss(beta=0.25)  # Huber Loss for out-of-distribution robustness

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    reg_target = torch.tensor(
        np.column_stack([mae_train / 3.0, mfe_train / 4.0]), dtype=torch.float32
    ).to(device)

    model.train()
    batch_size = 64
    n_samples = len(X_train)

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            indices = perm[i : i + batch_size]
            b_x = X_t[indices]
            b_y = y_t[indices]
            b_reg = reg_target[indices]

            optimizer.zero_grad()
            logits, reg_out = model(b_x)
            loss = cls_loss_fn(logits, b_y) + 0.5 * reg_loss_fn(reg_out, b_reg)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def run_training_pipeline(symbol: str = "EURUSD", days: int = 60):
    """Executes full training pipeline and saves 34-feature models."""
    logger.info(f"=== Starting Training Pipeline for {symbol} ({days} days) ===")
    df = fetch_candles_hybrid(symbol, days)
    if df.empty or len(df) < 500:
        logger.error(f"Insufficient data to train for {symbol}. Total rows: {len(df)}")
        return False

    dataset = prepare_dataset(df, max_holding=15, subsample=False)
    if dataset is None:
        logger.error("Failed to prepare dataset.")
        return False

    X = dataset["X"]
    y = dataset["y"]
    sample_weight = dataset.get("sample_weight")
    logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features (Expected: {len(FEATURE_NAMES)})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Train LightGBM with 34 features
    train_lgb = lgb.Dataset(X_scaled, label=y, weight=sample_weight, feature_name=FEATURE_NAMES)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    lgbm_model = lgb.train(params, train_lgb, num_boost_round=300)
    logger.info(f"LightGBM trained successfully for {symbol}.")

    # 2. Train ScalpNet with PyTorch (Huber Loss + Softplus)
    scalpnet = train_scalpnet_pytorch(
        X_train=X_scaled,
        y_train=y,
        mfe_train=dataset["mfe"],
        mae_train=dataset["mae"],
        n_features=X.shape[1],
        epochs=35
    )
    logger.info(f"ScalpNet trained successfully for {symbol}.")

    # 3. Save Models to models/ directory
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save Scaler as JSON
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    with open(models_dir / f"{symbol}_scaler.json", "w", encoding="utf-8") as f:
        json.dump(scaler_data, f, indent=2)

    # Save LightGBM
    lgbm_model.save_model(str(models_dir / f"{symbol}_lgbm.txt"))

    # Save ScalpNet weights
    torch.save(scalpnet.state_dict(), str(models_dir / f"{symbol}_scalpnet.pt"))

    logger.info(f"ALL MODELS SAVED TO {models_dir} for {symbol} (Input dim: {X.shape[1]})")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()

    run_training_pipeline(args.symbol, args.days)
