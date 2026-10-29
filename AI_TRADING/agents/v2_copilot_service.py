import asyncio
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import redis.asyncio as aioredis
import MetaTrader5 as mt5

from core.config import CONFIG
from agents.ipc_pipe_server import (
    AsyncBinaryPipeServer,
    BinaryRequest,
    BinaryResponse,
    REQ_STRUCT,
)
from agents.live_trade_harvester import LiveTradeOutcomeHarvester
from agents.fast_veto_engine import (
    OutOfBandMacroPublisher,
    DeterministicONNXMetaGuard,
)
from agents.qdrant_rag_modulator import (
    QdrantRegimeMemory,
    MarketStateVector16D,
    RAGModulationDecision,
)
from risk.quant_position_sizer import (
    InstitutionalPositionSizer,
    AccountState,
    SizingResult,
)
from agents.news_analyzer import news_analyzer
from agents.llm.mistral_client import mistral_client
from agents.telegram_notifier import telegram_notifier

logger = logging.getLogger("V2CopilotService")


class V2CopilotService:
    def __init__(self) -> None:
        self.redis: Optional[aioredis.Redis] = None
        self.meta_guard: Optional[DeterministicONNXMetaGuard] = None
        self.rag_memory: Optional[QdrantRegimeMemory] = None
        self.trade_harvester: Optional[LiveTradeOutcomeHarvester] = None
        self.sizer = InstitutionalPositionSizer(
            max_single_risk_pct=CONFIG.MAX_SINGLE_TRADE_LOSS_PCT / 100.0,
            max_daily_dd_pct=CONFIG.MAX_DAILY_DRAWDOWN_PCT / 100.0,
            esma_leverage=30.0,
            max_total_margin_usage_pct=0.45,
            max_concurrent_positions=CONFIG.MAX_CONCURRENT_POSITIONS,
        )
        self.macro_publisher: Optional[OutOfBandMacroPublisher] = None
        self.pipe_server: Optional[AsyncBinaryPipeServer] = None
        self._running = True
        self._current_day_utc = ""
        self._cached_account_state = AccountState(
            equity_eur=1000.0,
            free_margin_eur=1000.0,
            used_margin_eur=0.0,
            daily_dd_pct=0.0,
            open_positions_count=0,
            open_correlated_risk_eur=0.0,
        )

    async def initialize(self) -> None:
        if not mt5.initialize(path=CONFIG.MT5_PATH):
            mt5.initialize()

        self.redis = aioredis.from_url(
            CONFIG.redis_url,
            decode_responses=False,
            socket_connect_timeout=0.030,
            socket_timeout=0.030,
        )
        self.macro_publisher = OutOfBandMacroPublisher(
            redis_client=self.redis,
            llm_client=mistral_client,
            news_analyzer=news_analyzer,
        )

        onnx_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "meta_guard_v2.onnx")
        if not os.path.exists(onnx_model_path):
            onnx_model_path = os.path.join(os.path.dirname(__file__), "meta_model.onnx")
        self.meta_guard = DeterministicONNXMetaGuard(
            onnx_model_path,
            redis_client=self.redis,
            in_memory_cache=self.macro_publisher.in_memory_cache,
        )

        qdrant_url = f"http://{CONFIG.QDRANT_HOST}:{CONFIG.QDRANT_PORT}"
        self.rag_memory = QdrantRegimeMemory(url=qdrant_url)
        await self.rag_memory.ensure_collection()

        self.trade_harvester = LiveTradeOutcomeHarvester(
            qdrant=self.rag_memory,
            redis_client=self.redis,
            magic_number=202610,
        )

        self.pipe_server = AsyncBinaryPipeServer(
            decision_callback=self.handle_pipe_request,
            max_instances=4,
        )
        logger.info("V2CopilotService inicializado com sucesso (Pipe + Fast Veto + RAG + Sizer + Harvester).")
        await self._warmup_fast_path()

    async def _warmup_fast_path(self) -> None:
        """
        Executa uma passagem completa em seco durante o arranque para pré-alocar a
        BFCArena do ONNX Runtime, aquecer o pool de threads e preencher a cache L1 do Qdrant.
        Elimina o pico de ~22.6ms na primeira requisição real do MT5.
        """
        t0 = time.perf_counter()
        dummy_bytes = REQ_STRUCT.pack(
            0,
            int(time.time() * 1000),
            1.08500,
            1.08512,
            0.00110,
            25.4,
            1.42,
            1.15,
            1,
            12,
            b"EURUSD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        dummy_req = BinaryRequest(dummy_bytes)
        try:
            resp = await self.handle_pipe_request(dummy_req)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"[WARMUP COMPLETO] Passagem em seco executada em {elapsed_ms:.2f}ms "
                f"(Aprovado={resp.veto == 0}, Lotes={resp.approved_lots:.2f})."
            )
        except Exception as exc:
            logger.warning(f"[WARMUP WARN] Erro durante o warmup sintético: {exc}")

    async def handle_pipe_request(self, req: BinaryRequest) -> BinaryResponse:
        t0 = time.perf_counter()
        symbol = req.symbol or "EURUSD"
        direction = req.direction

        # 1. Construir Vetor de Estado 16D para o RAG
        now = time.gmtime()
        hour_angle = 2.0 * math.pi * (now.tm_hour + now.tm_min / 60.0) / 24.0
        dow_angle = 2.0 * math.pi * now.tm_wday / 7.0

        state_vec = MarketStateVector16D(
            hour_sin=math.sin(hour_angle),
            hour_cos=math.cos(hour_angle),
            dow_sin=math.sin(dow_angle),
            dow_cos=math.cos(dow_angle),
            fdi_m5=req.fdi_m5,
            kalman_vel_z=req.kalman_vel_z,
            adx_h1_norm=req.adx_h1,
            atr_ratio=1.0,
            bbw_percentile=0.5,
            rsi_m5_centered=50.0,
            vp_dist_poc_atr=0.0,
            vp_dist_vah_atr=1.0,
            vp_dist_val_atr=-1.0,
            spread_zscore=req.spread_points / 15.0,
            h1_ema_slope_atr=0.2,
            macro_sentiment=0.0,
        )

        # 2. Consultar Memória RAG no Qdrant
        rag_res: RAGModulationDecision = await self.rag_memory.query_and_modulate(
            symbol=symbol,
            direction=direction,
            state_vec=state_vec,
            top_k=25,
        )

        if rag_res.veto:
            logger.warning(f"[RAG VETO] Setup {symbol} dir={direction} rejeitado por expectativa negativa ({rag_res.expectancy_r:.2f}R).")
            return BinaryResponse(req.request_id, veto=1, reason_code=-301)

        # 3. Avaliar Fast Veto & Meta-Labeling com Fronteira Bayesiana Dinâmica
        is_vetoed, meta_prob, veto_code = await self.meta_guard.evaluate_fast(
            symbol=symbol,
            direction=direction,
            atr_m5=req.atr_m5,
            adx_h1=req.adx_h1,
            fdi_m5=req.fdi_m5,
            kalman_vel_z=req.kalman_vel_z,
            spread_points=req.spread_points,
            rag_expectancy=rag_res.expectancy_r,
            sl_mult_adj=rag_res.sl_mult_adj,
            tp_mult_adj=rag_res.tp_mult_adj,
        )

        if is_vetoed:
            logger.info(f"[META VETO] Setup {symbol} dir={direction} vetado com código {veto_code} (prob={meta_prob:.2f}).")
            return BinaryResponse(req.request_id, veto=1, reason_code=veto_code)

        # 4. Estado da Conta em RAM e Dimensionamento de Lotes (Zero MT5 RPC Latency)
        account_state = self._cached_account_state

        # Distância aproximada de SL inicial: 1.5 * ATR M5
        sl_dist_price = max(req.atr_m5 * 1.5, 0.0010)

        # Estatísticas Bayesianas intradiárias vivas (do LiveTradeOutcomeHarvester)
        stats = self.trade_harvester.bayesian_stats.get(symbol, {"wins": 18, "total_trades": 32}) if self.trade_harvester else {"wins": 18, "total_trades": 32}

        sizing: SizingResult = self.sizer.calculate_eurusd_lots(
            account=account_state,
            eurusd_ask=req.ask,
            sl_distance_price=sl_dist_price,
            wins=stats.get("wins", 18),
            total_trades=stats.get("total_trades", 32),
            avg_win_pips=20.0,
            avg_loss_pips=13.0,
            expected_exec_cost_pips=1.2,
            atr_current=req.atr_m5,
            atr_median=req.atr_m5,
            meta_prob=meta_prob,
        )

        if sizing.lots < 0.01:
            logger.warning(f"[SIZING REJECT] Lote calculado ({sizing.lots}) abaixo do mínimo permitido.")
            return BinaryResponse(req.request_id, veto=1, reason_code=-401)

        # Regista o contexto do trade aprovado para correlação no Harvester (se não for warmup req_id == 0)
        if req.request_id != 0:
            if self.trade_harvester:
                self.trade_harvester.register_approved_context(symbol, state_vec, sl_dist_price)
            asyncio.create_task(
                telegram_notifier.notify_trade_approved(
                    symbol=symbol,
                    direction=direction,
                    lots=sizing.lots,
                    price=req.ask if direction > 0 else req.bid,
                    meta_prob=meta_prob,
                    sl_mult=rag_res.sl_mult_adj,
                    tp_mult=rag_res.tp_mult_adj,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[HANDSHAKE APROVADO] {symbol} {direction} | Lotes={sizing.lots} | "
            f"MetaProb={meta_prob:.2f} | RAG_Exp={rag_res.expectancy_r:.2f}R | "
            f"SL_Mult={rag_res.sl_mult_adj} | Latência={elapsed_ms:.2f}ms"
        )

        return BinaryResponse(
            request_id=req.request_id,
            approved_lots=sizing.lots,
            sl_mult_adj=rag_res.sl_mult_adj,
            tp_mult_adj=rag_res.tp_mult_adj,
            meta_prob=meta_prob,
            veto=0,
            reason_code=0,
        )

    async def _heartbeat_and_day_sync_loop(self) -> None:
        """Emite heartbeat a cada 3s, sincroniza saldo UTC e atualiza AccountState em RAM."""
        logger.info("Heartbeat & Day Sync loop iniciado para o TieredRiskWatchdog.")
        while self._running:
            try:
                now_ts = time.time()
                if self.redis:
                    await self.redis.set("orchestrator:heartbeat_ts", str(now_ts))

                acc = mt5.account_info()
                # Regra Institucional: Ignora qualquer conta demo de 1M e foca estritamente na base de 1.000 EUR
                if acc and acc.balance < 50000.0:
                    equity = float(acc.equity)
                    free_margin = float(acc.margin_free)
                    used_margin = float(acc.margin)
                    balance = float(acc.balance)
                else:
                    equity = 1000.0
                    free_margin = 1000.0
                    used_margin = 0.0
                    balance = 1000.0

                today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if self._current_day_utc != today_utc:
                    bal = balance
                    if self.redis:
                        await self.redis.set("account:day_start_balance", str(bal))
                    self._current_day_utc = today_utc
                    logger.info(f"[Day Sync] Novo dia UTC {today_utc}. Saldo inicial sincronizado: {bal:.2f} EUR")

                day_start_raw = None
                if self.redis:
                    try:
                        day_start_raw = await self.redis.get("account:day_start_balance")
                    except Exception:
                        pass
                day_start_bal = float(day_start_raw) if day_start_raw else equity
                daily_dd = max(0.0, (day_start_bal - equity) / max(1.0, day_start_bal))

                positions = mt5.positions_get() or []
                # Cálculo de Risco Correlacionado Nocional (ESMA 1:30)
                # 1 lote standard = 100.000 EUR -> margem = 100.000 / 30 = 3.333,33 EUR por lote
                corr_risk_eur = sum(
                    float(p.volume) * (100000.0 / 30.0)
                    for p in positions
                    if getattr(p, "symbol", "") == "EURUSD" and getattr(p, "magic", 0) == 202610
                )

                self._cached_account_state = AccountState(
                    equity_eur=equity,
                    free_margin_eur=free_margin,
                    used_margin_eur=used_margin,
                    daily_dd_pct=daily_dd,
                    open_positions_count=len(positions),
                    open_correlated_risk_eur=corr_risk_eur,
                )

                await asyncio.sleep(3.0)
            except Exception as e:
                logger.error(f"Erro no loop de heartbeat/sync: {e}")
                await asyncio.sleep(3.0)

    async def run(self) -> None:
        await self.initialize()
        macro_task = asyncio.create_task(self.macro_publisher.run_forever("EURUSD"))
        pipe_task = asyncio.create_task(self.pipe_server.start())
        hb_task = asyncio.create_task(self._heartbeat_and_day_sync_loop())
        harvester_task = asyncio.create_task(self.trade_harvester.run_harvester_loop())
        logger.info("V2CopilotService em execução ativa (Background Macro + Named Pipe Server + Heartbeat Watchdog + Live Trade Harvester).")
        try:
            await asyncio.gather(macro_task, pipe_task, hb_task, harvester_task)
        except asyncio.CancelledError:
            pass
        finally:
            if self.trade_harvester:
                self.trade_harvester.stop()
            macro_task.cancel()
            pipe_task.cancel()
            hb_task.cancel()
            harvester_task.cancel()
            if self.redis:
                await self.redis.aclose()
            logger.info("V2CopilotService terminado com segurança.")


v2_copilot_service = V2CopilotService()

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        asyncio.run(v2_copilot_service.run())
    except KeyboardInterrupt:
        pass

