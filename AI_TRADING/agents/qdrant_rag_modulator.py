import asyncio
import math
import time
import numpy as np
from dataclasses import dataclass
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

COLLECTION_NAME = "ai_trading_regime_memory"


@dataclass(frozen=True)
class MarketStateVector16D:
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    fdi_m5: float
    kalman_vel_z: float
    adx_h1_norm: float
    atr_ratio: float
    bbw_percentile: float
    rsi_m5_centered: float
    vp_dist_poc_atr: float
    vp_dist_vah_atr: float
    vp_dist_val_atr: float
    spread_zscore: float
    h1_ema_slope_atr: float
    macro_sentiment: float

    def to_numpy(self) -> np.ndarray:
        raw = np.array([
            self.hour_sin, self.hour_cos,
            self.dow_sin, self.dow_cos,
            (self.fdi_m5 - 1.5) * 4.0,
            np.tanh(self.kalman_vel_z / 3.0),
            (self.adx_h1_norm - 25.0) / 15.0,
            np.tanh(self.atr_ratio - 1.0),
            (self.bbw_percentile - 0.5) * 2.0,
            (self.rsi_m5_centered - 50.0) / 25.0,
            np.tanh(self.vp_dist_poc_atr / 2.0),
            np.tanh(self.vp_dist_vah_atr / 2.0),
            np.tanh(self.vp_dist_val_atr / 2.0),
            np.tanh(self.spread_zscore / 3.0),
            np.tanh(self.h1_ema_slope_atr),
            np.clip(self.macro_sentiment, -1.0, 1.0),
        ], dtype=np.float32)
        return raw


@dataclass(frozen=True)
class RAGModulationDecision:
    veto: bool
    expectancy_r: float
    sl_mult_adj: float
    tp_mult_adj: float
    matched_neighbors: int


class QdrantRegimeMemory:
    def __init__(self, url: str = "http://localhost:6333") -> None:
        self.client = AsyncQdrantClient(url=url, timeout=0.030)
        self._cache: dict[str, tuple[float, RAGModulationDecision]] = {}

    async def ensure_collection(self) -> None:
        try:
            collections = await asyncio.wait_for(self.client.get_collections(), timeout=0.100)
            names = [c.name for c in collections.collections]
            if COLLECTION_NAME not in names:
                await self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config={
                        "quant_state": VectorParams(size=16, distance=Distance.EUCLID),
                        "semantic_context": VectorParams(size=384, distance=Distance.COSINE),
                    },
                )
        except Exception:
            # Não bloqueia se Qdrant estiver offline ou a inicializar
            pass

    async def query_and_modulate(
        self,
        symbol: str,
        direction: int,
        state_vec: MarketStateVector16D,
        top_k: int = 25,
        max_euclid_dist: float = 2.20,
    ) -> RAGModulationDecision:
        # 1. Fast RAM Cache Lookup (< 150ns)
        cache_key = f"{symbol}_{direction}_{round(state_vec.fdi_m5, 2)}_{round(state_vec.kalman_vel_z, 1)}"
        now_ts = time.time()
        if cache_key in self._cache:
            ts, cached_decision = self._cache[cache_key]
            if now_ts - ts < 30.0:
                return cached_decision

        query_vector = state_vec.to_numpy().tolist()

        try:
            search_coro = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=("quant_state", query_vector),
                query_filter=Filter(
                    must=[
                        FieldCondition(key="symbol", match=MatchValue(value=symbol)),
                        FieldCondition(key="direction", match=MatchValue(value=direction)),
                    ]
                ),
                limit=top_k,
                with_payload=True,
            )
            search_res = await asyncio.wait_for(search_coro, timeout=0.010)
        except Exception:
            # Fail-fast (<10ms): devolve neutro e cacheia para manter sub-milissegundos no caminho crítico
            decision = RAGModulationDecision(False, 0.0, 1.0, 1.0, 0)
            self._cache[cache_key] = (now_ts, decision)
            return decision

        # Filtra apenas vizinhos suficientemente próximos no espaço 16D
        valid_hits = [hit for hit in search_res if hit.score <= max_euclid_dist and hit.payload]
        if len(valid_hits) < 8:
            # Amostra histórica local insuficiente -> Neutro (sem veto, multiplicadores 1.0)
            decision = RAGModulationDecision(False, 0.0, 1.0, 1.0, len(valid_hits))
            self._cache[cache_key] = (now_ts, decision)
            return decision

        weights = []
        pnls_r = []
        maes_r = []
        mfes_r = []

        for hit in valid_hits:
            w = 1.0 / (float(hit.score) + 0.05)
            weights.append(w)
            pnls_r.append(float(hit.payload.get("realized_r", 0.0)))
            maes_r.append(float(hit.payload.get("mae_r", 1.0)))  # Excursão adversa em múltiplos de R inicial
            mfes_r.append(float(hit.payload.get("mfe_r", 1.5)))  # Excursão favorável em múltiplos de R inicial

        w_arr = np.array(weights, dtype=np.float64)
        w_norm = w_arr / np.sum(w_arr)

        expectancy_r = float(np.sum(w_norm * np.array(pnls_r)))
        if expectancy_r < -0.15:
            # Estados históricos idênticos resultaram sistematicamente em perda
            return RAGModulationDecision(True, expectancy_r, 1.0, 1.0, len(valid_hits))

        # Filtra vencedores para calibrar o Stop Loss ótimo sem ser violinado
        winning_maes = [m for m, r in zip(maes_r, pnls_r) if r > 0.2]
        if len(winning_maes) >= 4:
            q80_win_mae = float(np.percentile(winning_maes, 80))
            # Ajusta o SL para cobrir o percentil 80 de excursão adversa dos vencedores + 15% buffer
            sl_mult_adj = float(np.clip(q80_win_mae * 1.15, 0.75, 1.35))
        else:
            sl_mult_adj = 1.0

        q65_mfe = float(np.percentile(mfes_r, 65))
        tp_mult_adj = float(np.clip(q65_mfe * 0.90, 0.80, 1.60))

        decision = RAGModulationDecision(
            veto=False,
            expectancy_r=round(expectancy_r, 4),
            sl_mult_adj=round(sl_mult_adj, 3),
            tp_mult_adj=round(tp_mult_adj, 3),
            matched_neighbors=len(valid_hits),
        )
        self._cache[cache_key] = (now_ts, decision)
        return decision

    async def record_closed_trade_outcome(
        self,
        trade_uuid: str,
        symbol: str,
        direction: int,
        state_vec: MarketStateVector16D,
        semantic_vec: list[float],
        realized_r: float,
        mae_r: float,
        mfe_r: float,
    ) -> None:
        point = PointStruct(
            id=trade_uuid,
            vector={
                "quant_state": state_vec.to_numpy().tolist(),
                "semantic_context": semantic_vec,
            },
            payload={
                "symbol": symbol,
                "direction": direction,
                "realized_r": realized_r,
                "mae_r": mae_r,
                "mfe_r": mfe_r,
            },
        )
        try:
            await self.client.upsert(collection_name=COLLECTION_NAME, points=[point])
        except Exception:
            pass
