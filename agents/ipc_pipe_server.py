import asyncio
import struct
import time
import logging
from typing import Callable, Awaitable
import win32pipe
import win32file
import pywintypes

logger = logging.getLogger("IPC_BinaryPipe")

PIPE_NAME = r"\\.\pipe\AI_TRADING_TWIN_V2"

# < = Little-Endian | Q = uint64 | q = int64 | 6d = 6 float64 | 2i = 2 int32 | 16s = char[16] (88 bytes)
REQ_STRUCT = struct.Struct("<Qq6d2i16s")
# < = Little-Endian | Q = uint64 | q = int64 | 4d = 4 float64 | 2i = 2 int32 (56 bytes)
RES_STRUCT = struct.Struct("<Qq4d2i")


class BinaryRequest:
    __slots__ = (
        "request_id", "timestamp_msc", "bid", "ask", "atr_m5", "adx_h1",
        "fdi_m5", "kalman_vel_z", "direction", "spread_points", "symbol"
    )

    def __init__(self, raw_bytes: bytes) -> None:
        unpacked = REQ_STRUCT.unpack(raw_bytes)
        self.request_id: int = unpacked[0]
        self.timestamp_msc: int = unpacked[1]
        self.bid: float = unpacked[2]
        self.ask: float = unpacked[3]
        self.atr_m5: float = unpacked[4]
        self.adx_h1: float = unpacked[5]
        self.fdi_m5: float = unpacked[6]
        self.kalman_vel_z: float = unpacked[7]
        self.direction: int = unpacked[8]
        self.spread_points: int = unpacked[9]
        self.symbol: str = unpacked[10].split(b"\x00", 1)[0].decode("utf-8", errors="ignore")


class BinaryResponse:
    __slots__ = (
        "request_id", "timestamp_msc", "approved_lots",
        "sl_mult_adj", "tp_mult_adj", "meta_prob", "veto", "reason_code"
    )

    def __init__(
        self,
        request_id: int,
        approved_lots: float = 0.0,
        sl_mult_adj: float = 1.0,
        tp_mult_adj: float = 1.0,
        meta_prob: float = 0.0,
        veto: int = 1,
        reason_code: int = 0,
    ) -> None:
        self.request_id = request_id
        self.timestamp_msc = int(time.time() * 1000)
        self.approved_lots = approved_lots
        self.sl_mult_adj = sl_mult_adj
        self.tp_mult_adj = tp_mult_adj
        self.meta_prob = meta_prob
        self.veto = veto
        self.reason_code = reason_code

    def pack(self) -> bytes:
        return RES_STRUCT.pack(
            self.request_id,
            self.timestamp_msc,
            self.approved_lots,
            self.sl_mult_adj,
            self.tp_mult_adj,
            self.meta_prob,
            self.veto,
            self.reason_code,
        )


from concurrent.futures import ThreadPoolExecutor

class AsyncBinaryPipeServer:
    def __init__(
        self,
        decision_callback: Callable[[BinaryRequest], Awaitable[BinaryResponse]],
        max_instances: int = 4,
    ) -> None:
        self.decision_callback = decision_callback
        self.max_instances = max_instances
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_instances + 2,
            thread_name_prefix="NamedPipeWorker"
        )
        self._running = True

    def stop(self) -> None:
        self._running = False

    def _create_pipe_instance(self) -> int:
        return win32pipe.CreateNamedPipe(
            PIPE_NAME,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
            win32pipe.PIPE_UNLIMITED_INSTANCES,
            RES_STRUCT.size,
            REQ_STRUCT.size,
            0,
            None,
        )

    def _connect_named_pipe(self, pipe_handle: int) -> None:
        try:
            win32pipe.ConnectNamedPipe(pipe_handle, None)
        except pywintypes.error as e:
            # 535 = ERROR_PIPE_CONNECTED: Cliente já se conectou entre a criação e a chamada
            if e.winerror != 535:
                raise

    async def _worker_loop(self, worker_id: int) -> None:
        loop = asyncio.get_running_loop()
        pipe_handle = await loop.run_in_executor(self.executor, self._create_pipe_instance)
        try:
            while self._running:
                try:
                    await loop.run_in_executor(self.executor, self._connect_named_pipe, pipe_handle)
                    hr, raw_data = await loop.run_in_executor(
                        self.executor, win32file.ReadFile, pipe_handle, REQ_STRUCT.size
                    )
                    if len(raw_data) == REQ_STRUCT.size:
                        req = BinaryRequest(raw_data)
                        try:
                            # Orçamento de 500ms na fast-path (mais de 6x abaixo dos 3000ms do MQL5)
                            res = await asyncio.wait_for(self.decision_callback(req), timeout=0.500)
                        except Exception as exc:
                            logger.error(f"Erro no callback de decisão: {exc}")
                            res = BinaryResponse(req.request_id, veto=1, reason_code=-500)

                        payload = res.pack()
                        await loop.run_in_executor(
                            self.executor, win32file.WriteFile, pipe_handle, payload
                        )
                        # Aguarda que o cliente MQL5 feche o seu lado apos ler os 56 bytes (gera erro 109 imediato)
                        try:
                            await loop.run_in_executor(
                                self.executor, win32file.ReadFile, pipe_handle, 1
                            )
                        except pywintypes.error:
                            pass
                    else:
                        logger.warning(f"Worker {worker_id}: Pacote truncado ({len(raw_data)} bytes).")
                except pywintypes.error as e:
                    if e.winerror not in (109, 232, 233, 535):
                        logger.warning(f"Worker {worker_id} pipe error: {e}")
                finally:
                    # Desconecta apenas a sessao do cliente, mantendo o pipe_handle vivo no kernel!
                    try:
                        await loop.run_in_executor(
                            self.executor, win32pipe.DisconnectNamedPipe, pipe_handle
                        )
                    except Exception:
                        pass
        finally:
            try:
                await loop.run_in_executor(self.executor, win32file.CloseHandle, pipe_handle)
            except Exception:
                pass

    async def start(self) -> None:
        tasks = [asyncio.create_task(self._worker_loop(i)) for i in range(self.max_instances)]
        try:
            await asyncio.gather(*tasks)
        finally:
            self.executor.shutdown(wait=False)

