import asyncio
import time
import json
import os
import logging
from typing import Optional
import numpy as np
import redis.asyncio as aioredis

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logger = logging.getLogger("FastVetoEngine")


class OutOfBandMacroPublisher:
    """
    Corre em background fora do caminho crítico. Atualiza o estado de risco macro
    no DragonflyDB e na cache in-memory com predição antecipada antes do fecho de cada vela M5.
    """
    def __init__(self, redis_client: Optional[aioredis.Redis] = None, llm_client=None, news_analyzer=None) -> None:
        self.redis = redis_client
        self.llm = llm_client
        self.news = news_analyzer
        self._running = True
        self.in_memory_cache: dict[str, dict] = {}

    async def run_forever(self, symbol: str = "EURUSD") -> None:
        logger.info(f"OutOfBandMacroPublisher iniciado para {symbol}.")
        while self._running:
            try:
                # Sincroniza para correr periodicamente e antes do fecho de cada vela M5
                now = time.time()
                seconds_in_m5 = now % 300
                sleep_sec = 15.0 if seconds_in_m5 < 280 else max(0.5, 295.0 - seconds_in_m5)

                news_summary = {}
                if self.news and hasattr(self.news, "get_cached_or_fetch"):
                    news_summary = await self.news.get_cached_or_fetch(symbol)
                elif self.news and hasattr(self.news, "analyze_symbol_news"):
                    news_summary = await self.news.analyze_symbol_news(symbol)

                macro_eval = {}
                if self.llm and hasattr(self.llm, "evaluate_tail_risk_async"):
                    macro_eval = await self.llm.evaluate_tail_risk_async(symbol, news_summary)

                state_payload = {
                    "veto_long": int(macro_eval.get("veto_long", False)),
                    "veto_short": int(macro_eval.get("veto_short", False)),
                    "sentiment_score": float(macro_eval.get("sentiment_score", 0.0)),
                    "event_imminent": int(macro_eval.get("high_impact_within_15m", False)),
                    "updated_at_ms": int(time.time() * 1000),
                }

                # 1. Atualização Instantânea em RAM (Zero-Socket Fast Path < 100ns)
                self.in_memory_cache[symbol] = state_payload

                # 2. Persistência Externa em Redis/Dragonfly para o Watchdog
                if self.redis:
                    try:
                        key = f"fast_veto:macro:{symbol}"
                        await self.redis.hset(key, mapping=state_payload)
                        await self.redis.expire(key, 90)
                    except Exception:
                        pass

                await asyncio.sleep(sleep_sec)
            except Exception as exc:
                logger.error(f"Erro no OutOfBandMacroPublisher: {exc}")
                await asyncio.sleep(5.0)


class DeterministicONNXMetaGuard:
    """
    Substitui o LLM na critical path por um classificador LightGBM/XGBoost calibrado
    para ONNX com Fronteira Bayesiana Dinâmica. Latência de inferência < 0.25 ms.
    """
    def __init__(
        self,
        onnx_model_path: Optional[str],
        redis_client: Optional[aioredis.Redis] = None,
        in_memory_cache: Optional[dict] = None,
    ) -> None:
        self.redis = redis_client
        self.in_memory_cache = in_memory_cache
        self.session = None
        self.input_name = None

        if HAS_ONNX and onnx_model_path and os.path.exists(onnx_model_path):
            try:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(
                    onnx_model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                logger.info(f"Modelo ONNX carregado com sucesso a partir de {onnx_model_path}")
            except Exception as e:
                logger.warning(f"Falha ao carregar modelo ONNX ({e}). A usar fallback meta-heurístico.")
        else:
            logger.info("Modo Meta-Heurístico calibrado ativo (ONNX offline/em treino).")

    @staticmethod
    def compute_dynamic_veto_threshold(
        fdi_m5: float,
        atr_m5: float,
        spread_points: int,
        sl_mult_adj: float = 1.0,
        tp_mult_adj: float = 1.0,
        rag_expectancy: float = 0.0,
        base_rr: float = 1.466,  # 2.2 ATR TP / 1.5 ATR SL
    ) -> float:
        """
        Calcula a fronteira de decisão ótima p_crit(t) balanceando custo microestrutural,
        regime caótico de Sevcik e expectativa local da memória RAG.
        """
        spread_price = spread_points * 0.00001
        base_sl_price = max(1.5 * atr_m5, 1e-5)
        cost_ratio = spread_price / base_sl_price

        b_net = max(0.5, (base_rr * tp_mult_adj - cost_ratio) / (sl_mult_adj + cost_ratio))
        be_threshold = (1.0 + 0.08) / (1.0 + b_net)

        fdi_penalty = 0.18 * max(0.0, fdi_m5 - 1.42)
        rag_bonus = 0.05 * float(np.tanh(rag_expectancy))

        dynamic_p_crit = be_threshold + fdi_penalty - rag_bonus
        return float(min(0.64, max(0.38, dynamic_p_crit)))

    async def evaluate_fast(
        self,
        symbol: str,
        direction: int,
        atr_m5: float,
        adx_h1: float,
        fdi_m5: float,
        kalman_vel_z: float,
        spread_points: int,
        rag_expectancy: float,
        sl_mult_adj: float = 1.0,
        tp_mult_adj: float = 1.0,
    ) -> tuple[bool, float, int]:
        """
        Retorna: (is_vetoed, meta_probability, reason_code)
        """
        # 1. Tenta leitura ultrarrápida da RAM in-process (< 100ns)
        macro_state = self.in_memory_cache.get(symbol) if self.in_memory_cache else None

        # 2. Fallback para Redis se não existir em RAM
        if not macro_state and self.redis:
            try:
                macro_state = await self.redis.hgetall(f"fast_veto:macro:{symbol}")
            except Exception:
                macro_state = {}

        if not macro_state:
            sentiment = 0.0
            event_imminent = 0
            veto_long = 0
            veto_short = 0
        else:
            event_imminent = int(macro_state.get(b"event_imminent", macro_state.get("event_imminent", 0)))
            veto_long = int(macro_state.get(b"veto_long", macro_state.get("veto_long", 0)))
            veto_short = int(macro_state.get(b"veto_short", macro_state.get("veto_short", 0)))
            sentiment_val = macro_state.get(b"sentiment_score", macro_state.get("sentiment_score", 0.0))
            sentiment = float(sentiment_val)

        if event_imminent == 1:
            return True, 0.0, -202  # Evento NFP/CPI/FOMC iminente

        if direction == 1 and veto_long == 1:
            return True, 0.0, -203
        if direction == -1 and veto_short == 1:
            return True, 0.0, -204

        # 3. Cálculo da Fronteira Bayesiana Dinâmica p_crit(t)
        p_crit = self.compute_dynamic_veto_threshold(
            fdi_m5=fdi_m5,
            atr_m5=atr_m5,
            spread_points=spread_points,
            sl_mult_adj=sl_mult_adj,
            tp_mult_adj=tp_mult_adj,
            rag_expectancy=rag_expectancy,
        )

        # 4. Inferência ONNX ultra-rápida
        if self.session and self.input_name:
            now_struct = time.gmtime()
            hour_angle = 2.0 * np.pi * (now_struct.tm_hour + now_struct.tm_min / 60.0) / 24.0

            features = np.array(
                [[
                    float(direction),
                    atr_m5 * 10000.0,        # ATR em pips
                    adx_h1,
                    fdi_m5,
                    kalman_vel_z,
                    float(spread_points),
                    sentiment * float(direction),  # Alinhamento de sentimento com direção
                    rag_expectancy,
                    np.sin(hour_angle),
                    np.cos(hour_angle),
                ]],
                dtype=np.float32,
            )

            outputs = self.session.run(None, {self.input_name: features})
            if len(outputs) > 1 and isinstance(outputs[1], np.ndarray):
                prob_win = float(outputs[1][0][1])
            elif len(outputs) > 1 and isinstance(outputs[1], list):
                prob_win = float(outputs[1][0].get(1, 0.5))
            else:
                prob_win = float(outputs[0][0])

            if prob_win < p_crit:
                return True, prob_win, -205

            return False, prob_win, 0

        # Heurística analítica calibrada (quando ONNX não está presente)
        base_prob = 0.53
        if fdi_m5 > 1.54:
            base_prob -= 0.06
        elif fdi_m5 < 1.42:
            base_prob += 0.04

        if (direction == 1 and kalman_vel_z > 1.0) or (direction == -1 and kalman_vel_z < -1.0):
            base_prob += 0.03
        elif (direction == 1 and kalman_vel_z < -1.0) or (direction == -1 and kalman_vel_z > 1.0):
            base_prob -= 0.05

        base_prob += float(np.clip(rag_expectancy * 0.1, -0.08, 0.08))

        if base_prob < p_crit:
            return True, round(base_prob, 3), -206

        return False, round(base_prob, 3), 0
