import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import MetaTrader5 as mt5

from agents.qdrant_rag_modulator import MarketStateVector16D, QdrantRegimeMemory

logger = logging.getLogger("LiveTradeOutcomeHarvester")


class LiveTradeOutcomeHarvester:
    """
    Monitoriza negócios fechados no MT5 (DEAL_ENTRY_OUT), calcula Realized_R, MAE_R e MFE_R
    sobre os ticks reais da duração da operação e alimenta o Qdrant 16D + contadores Bayesianos.
    """
    def __init__(self, qdrant: QdrantRegimeMemory, redis_client=None, magic_number: int = 202610) -> None:
        self.qdrant = qdrant
        self.redis = redis_client
        self.magic_number = magic_number
        self._pending_entries: dict[int, tuple[MarketStateVector16D, float]] = {}
        self._last_approved_context: dict[str, tuple[MarketStateVector16D, float, float]] = {}
        self._processed_exit_deals: set[int] = set()
        self._last_check_ts: float = time.time() - 300.0
        self._running: bool = True
        
        # Estatísticas Bayesianas intradiárias em RAM (Zero-latency fallback)
        self.bayesian_stats: dict[str, dict[str, int]] = {
            "EURUSD": {"wins": 18, "total_trades": 32}
        }

    def register_approved_context(
        self, symbol: str, state_vec: MarketStateVector16D, sl_distance_price: float
    ) -> None:
        """Regista o contexto do último sinal aprovado para mapear à posição quando abrir."""
        self._last_approved_context[symbol] = (state_vec, max(sl_distance_price, 1e-5), time.time())

    def register_entry_context(
        self, position_id: int, state_vec: MarketStateVector16D, sl_distance_price: float
    ) -> None:
        """Associa explicitamente uma posição aberta ao seu vetor 16D e SL inicial."""
        self._pending_entries[position_id] = (state_vec, max(sl_distance_price, 1e-5))

    async def run_harvester_loop(self) -> None:
        """Ciclo periódico assíncrono que corre em background."""
        logger.info("LiveTradeOutcomeHarvester em execução ativa (Polling Deals MT5 a cada 15s).")
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                await asyncio.sleep(15.0)
                await loop.run_in_executor(None, self._poll_and_process_deals_sync, loop)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"[Harvester] Erro no ciclo de recolha: {exc}")

    def stop(self) -> None:
        self._running = False

    def _poll_and_process_deals_sync(self, loop: asyncio.AbstractEventLoop) -> None:
        # 1. Mapeia posições recém-abertas ao contexto aprovado mais recente
        try:
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    if pos.magic == self.magic_number and pos.ticket not in self._pending_entries:
                        sym = pos.symbol
                        if sym in self._last_approved_context:
                            s_vec, sl_d, t_appr = self._last_approved_context[sym]
                            if time.time() - t_appr < 180.0:  # Válido até 3 minutos pós-aprovação
                                self.register_entry_context(pos.ticket, s_vec, sl_d)
        except Exception:
            pass

        # 2. Interroga o histórico de negócios (Deals) fechados
        now_ts = time.time()
        try:
            deals = mt5.history_deals_get(self._last_check_ts - 300.0, now_ts + 60.0)
        except Exception:
            deals = None

        self._last_check_ts = now_ts
        if not deals:
            return

        for deal in deals:
            if deal.ticket in self._processed_exit_deals:
                continue
            if deal.magic != self.magic_number or deal.entry != mt5.DEAL_ENTRY_OUT:
                continue

            self._processed_exit_deals.add(deal.ticket)
            pos_id = deal.position_id

            # Procura o negócio de abertura correspondente (DEAL_ENTRY_IN)
            try:
                pos_deals = mt5.history_deals_get(position=pos_id)
            except Exception:
                pos_deals = None

            if not pos_deals or len(pos_deals) < 2:
                continue

            entry_deal = next((d for d in pos_deals if d.entry == mt5.DEAL_ENTRY_IN), None)
            if entry_deal is None:
                continue

            direction = 1 if entry_deal.type == mt5.DEAL_TYPE_BUY else -1
            entry_price = float(entry_deal.price)
            exit_price = float(deal.price)

            # Recupera o vetor de estado guardado na abertura
            ctx = self._pending_entries.pop(pos_id, None)
            sl_dist = ctx[1] if ctx else 0.0015
            state_vec = ctx[0] if ctx else None
            if state_vec is None:
                continue

            realized_r = ((exit_price - entry_price) * direction) / sl_dist

            # Extrai os ticks reais entre abertura e fecho para calcular MAE_R e MFE_R exatos
            t_from = datetime.fromtimestamp(entry_deal.time, tz=timezone.utc)
            t_to = datetime.fromtimestamp(deal.time + 1, tz=timezone.utc)
            try:
                ticks = mt5.copy_ticks_range(deal.symbol, t_from, t_to, mt5.COPY_TICKS_INFO)
            except Exception:
                ticks = None

            if ticks is not None and len(ticks) > 2:
                bids = ticks["bid"].astype(np.float64)
                asks = ticks["ask"].astype(np.float64)
                if direction == 1:
                    mfe_r = float(max(0.0, np.max(bids - entry_price)) / sl_dist)
                    mae_r = float(max(0.0, np.max(entry_price - bids)) / sl_dist)
                else:
                    mfe_r = float(max(0.0, np.max(entry_price - asks)) / sl_dist)
                    mae_r = float(max(0.0, np.max(asks - entry_price)) / sl_dist)
            else:
                mfe_r = max(0.0, realized_r)
                mae_r = max(0.0, -realized_r)

            # Agenda gravação assíncrona no Qdrant e atualização Bayesiana
            asyncio.run_coroutine_threadsafe(
                self._persist_outcome_async(
                    symbol=deal.symbol,
                    direction=direction,
                    state_vec=state_vec,
                    realized_r=round(realized_r, 3),
                    mae_r=round(mae_r, 3),
                    mfe_r=round(mfe_r, 3),
                ),
                loop,
            )

    async def _persist_outcome_async(
        self,
        symbol: str,
        direction: int,
        state_vec: MarketStateVector16D,
        realized_r: float,
        mae_r: float,
        mfe_r: float,
    ) -> None:
        # Atualiza contadores Bayesianos em RAM
        if symbol not in self.bayesian_stats:
            self.bayesian_stats[symbol] = {"wins": 18, "total_trades": 32}
        self.bayesian_stats[symbol]["total_trades"] += 1
        if realized_r > 0.05:
            self.bayesian_stats[symbol]["wins"] += 1

        try:
            await self.qdrant.record_closed_trade_outcome(
                trade_uuid=str(uuid.uuid4()),
                symbol=symbol,
                direction=direction,
                state_vec=state_vec,
                semantic_vec=[0.0] * 384,
                realized_r=realized_r,
                mae_r=mae_r,
                mfe_r=mfe_r,
            )
        except Exception as e:
            logger.warning(f"[Harvester] Falha ao persistir no Qdrant: {e}")

        if self.redis is not None:
            try:
                await self.redis.hincrby(f"stats:bayesian:{symbol}", "total_trades", 1)
                if realized_r > 0.05:
                    await self.redis.hincrby(f"stats:bayesian:{symbol}", "wins", 1)
            except Exception:
                pass

        logger.info(
            f"[HARVESTER RAG] Trade fechado ingerido no Qdrant: {symbol} dir={direction} "
            f"| R={realized_r:+.2f}R | MAE={mae_r:.2f}R | MFE={mfe_r:.2f}R | "
            f"Bayes Wins={self.bayesian_stats[symbol]['wins']}/{self.bayesian_stats[symbol]['total_trades']}"
        )
