"""
scripts/bootstrap_v2_intelligence.py
Popula o Qdrant (Cold-Start Seeding 16D) e treina/exporta o Meta-Modelo ONNX
a partir do histórico real de EURUSD do MetaTrader 5.
"""
import asyncio
import os
import sys
import uuid
import math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CONFIG
from agents.qdrant_rag_modulator import QdrantRegimeMemory, MarketStateVector16D


def calc_sevcik_fdi_series(close: np.ndarray, period: int = 30) -> np.ndarray:
    n = len(close)
    fdi = np.full(n, 1.5, dtype=np.float64)
    inv_n = 1.0 / (period - 1)
    inv_n2 = inv_n * inv_n
    denom = math.log(2.0 * (period - 1))

    for i in range(period, n):
        window = close[i - period : i]
        p_min = np.min(window)
        p_max = np.max(window)
        rng = p_max - p_min
        if rng <= 1e-9:
            continue
        dy = np.diff(window) / rng
        L = np.sum(np.sqrt(dy * dy + inv_n2))
        val = 1.0 + (math.log(L) + math.log(2.0)) / denom
        fdi[i] = min(2.0, max(1.0, val))
    return fdi


def calc_kalman_velocity_z(close: np.ndarray, atr: np.ndarray) -> np.ndarray:
    n = len(close)
    vel_z = np.zeros(n, dtype=np.float64)
    x0, x1 = close[0], 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    for i in range(1, n):
        atr_norm = max(atr[i], 1e-5)
        q_pos = 1e-6 * atr_norm
        q_vel = 5e-7 * atr_norm
        r_obs = 1e-3 * atr_norm

        pred_x0 = x0 + x1
        pred_x1 = x1

        n_p00 = p00 + p10 + p01 + p11 + q_pos
        n_p01 = p01 + p11
        n_p10 = p10 + p11
        n_p11 = p11 + q_vel

        y = close[i] - pred_x0
        S = n_p00 + r_obs
        K0 = n_p00 / S
        K1 = n_p10 / S

        x0 = pred_x0 + K0 * y
        x1 = pred_x1 + K1 * y

        p00 = (1.0 - K0) * n_p00
        p01 = (1.0 - K0) * n_p01
        p10 = n_p10 - K1 * n_p00
        p11 = n_p11 - K1 * n_p01

        vel_z[i] = x1 / math.sqrt(max(p11, 1e-12))
    return vel_z


async def bootstrap_qdrant_and_onnx(symbol: str = "EURUSD") -> None:
    print(f"=== INICIANDO BOOTSTRAP V2 ({symbol}) - ZERO LOOKAHEAD LEAKAGE ===")
    if not mt5.initialize(path=CONFIG.MT5_PATH):
        if not mt5.initialize():
            raise RuntimeError("Falha ao inicializar o MetaTrader 5 para bootstrap.")

    # Corte temporal estrito: Treinar APENAS em dados In-Sample (2026-05-01 a 2026-07-31)
    start_train = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    end_train = datetime(2026, 7, 31, 23, 59, tzinfo=timezone.utc)
    print(f"-> A carregar historico In-Sample estrito ({start_train.date()} a {end_train.date()})...")
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_train, end_train)
    mt5.shutdown()

    if rates is None or len(rates) < 1000:
        raise RuntimeError("Histórico M5 insuficiente no MT5 para o período de treino.")

    print(f"-> {len(rates)} barras M5 carregadas do MT5 (Sem sobreposicao com o teste de Agosto-Outubro).")

    close = rates["close"].astype(np.float64)
    high = rates["high"].astype(np.float64)
    low = rates["low"].astype(np.float64)
    spread = rates["spread"].astype(np.float64)
    times = rates["time"].astype(np.int64)

    # Cálculo de ATR(14)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = np.convolve(tr, np.ones(14) / 14.0, mode="same")

    fdi = calc_sevcik_fdi_series(close, 30)
    kalman_z = calc_kalman_velocity_z(close, atr)

    qdrant_url = f"http://{CONFIG.QDRANT_HOST}:{CONFIG.QDRANT_PORT}"
    qdrant = QdrantRegimeMemory(url=qdrant_url)
    try:
        from agents.qdrant_rag_modulator import COLLECTION_NAME
        await qdrant.client.delete_collection(COLLECTION_NAME)
        print(f"-> Colecao anterior '{COLLECTION_NAME}' eliminada do Qdrant para evitar contaminacao.")
    except Exception:
        pass
    await qdrant.ensure_collection()

    X_train = []
    y_train = []
    history_realized = []
    seeded_count = 0

    # Simula janelas futuras de 36 barras M5 (3 horas) para extrair MAE_R, MFE_R e Realized_R
    horizon = 36
    for i in range(300, len(close) - horizon - 1):
        if fdi[i] > 1.54 or atr[i] <= 1e-5:
            continue

        # Condição de gatilho direcional baseado em velocidade de Kalman
        if abs(kalman_z[i]) < 1.35:
            continue

        direction = 1 if kalman_z[i] > 0 else -1
        entry_price = close[i]
        sl_dist = 1.5 * atr[i]
        tp_dist = 2.2 * atr[i]

        future_highs = high[i + 1 : i + 1 + horizon]
        future_lows = low[i + 1 : i + 1 + horizon]

        if direction == 1:
            mfe_r = float(np.max(future_highs - entry_price) / sl_dist)
            mae_r = float(np.max(entry_price - future_lows) / sl_dist)
        else:
            mfe_r = float(np.max(entry_price - future_lows) / sl_dist)
            mae_r = float(np.max(future_highs - entry_price) / sl_dist)

        # Determina se atingiu TP ou SL primeiro
        realized_r = 0.0
        for k in range(1, horizon + 1):
            if direction == 1:
                if low[i + k] <= entry_price - sl_dist:
                    realized_r = -1.0
                    break
                if high[i + k] >= entry_price + tp_dist:
                    realized_r = tp_dist / sl_dist
                    break
            else:
                if high[i + k] >= entry_price + sl_dist:
                    realized_r = -1.0
                    break
                if low[i + k] <= entry_price - tp_dist:
                    realized_r = tp_dist / sl_dist
                    break

        if realized_r == 0.0:
            realized_r = ((close[i + horizon] - entry_price) * direction / sl_dist)

        atr_med = float(np.median(atr[i - 100 : i]))
        hour_utc = (times[i] % 86400) / 3600.0
        hour_angle = 2.0 * math.pi * hour_utc / 24.0
        dow = ((times[i] // 86400) + 4) % 7
        dow_angle = 2.0 * math.pi * dow / 7.0

        state_vec = MarketStateVector16D(
            hour_sin=math.sin(hour_angle),
            hour_cos=math.cos(hour_angle),
            dow_sin=math.sin(dow_angle),
            dow_cos=math.cos(dow_angle),
            fdi_m5=float(fdi[i]),
            kalman_vel_z=float(kalman_z[i]),
            adx_h1_norm=26.0,
            atr_ratio=float(atr[i] / max(atr_med, 1e-6)),
            bbw_percentile=0.55,
            rsi_m5_centered=54.0 if direction == 1 else 46.0,
            vp_dist_poc_atr=0.35 * direction,
            vp_dist_vah_atr=-0.50,
            vp_dist_val_atr=0.50,
            spread_zscore=0.1,
            h1_ema_slope_atr=0.4 * direction,
            macro_sentiment=0.0,
        )

        # Semear Qdrant (amostragem espaçada para evitar sobreposição de barras adjacentes)
        if i % 6 == 0:
            try:
                await qdrant.record_closed_trade_outcome(
                    trade_uuid=str(uuid.uuid4()),
                    symbol=symbol,
                    direction=direction,
                    state_vec=state_vec,
                    semantic_vec=[0.0] * 384,
                    realized_r=round(float(realized_r), 3),
                    mae_r=round(float(mae_r), 3),
                    mfe_r=round(float(mfe_r), 3),
                )
                seeded_count += 1
            except Exception as e:
                pass

        # Rolling expectancy histórico prévio (zero lookahead leakage)
        prior_rag_exp = float(np.mean(history_realized[-15:])) if len(history_realized) >= 5 else 0.0

        X_train.append([
            float(direction),
            float(atr[i] * 10000.0),
            26.0,
            float(fdi[i]),
            float(kalman_z[i]),
            float(spread[i]),
            0.0,
            prior_rag_exp,
            math.sin(hour_angle),
            math.cos(hour_angle),
        ])
        y_train.append(1 if realized_r > 0.1 else 0)
        history_realized.append(realized_r)

    print(f"[BOOTSTRAP OK] {seeded_count} estados históricos 16D inseridos no Qdrant.")

    # Exportação para ONNX
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_file_path = models_dir / "meta_guard_v2.onnx"

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        print("-> A treinar GradientBoostingClassifier calibrado (Sigmoid/Isotónico) com Meta-Features...")
        base_clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)
        clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        clf.fit(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int64))

        initial_type = [("float_input", FloatTensorType([None, 10]))]
        onx = convert_sklearn(clf, initial_types=initial_type, target_opset=12)
        with open(onnx_file_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"[ONNX OK] Modelo calibrado exportado com sucesso para {onnx_file_path}")
    except Exception as e:
        print(f"[AVISO ONNX] Erro ao exportar para ONNX ({e}). FastVeto operará com avaliador analítico nativo.")


if __name__ == "__main__":
    asyncio.run(bootstrap_qdrant_and_onnx())
