"""
AI Trading System — Windows Named Pipe IPC Server
Provides sub-millisecond (< 1ms) bidirectional IPC between MQL5 and Python.
"""
import asyncio
import json
import logging
import win32pipe
import win32file
import pywintypes

logger = logging.getLogger("ipc_pipe_server")

PIPE_NAME = r"\\.\pipe\AI_TRADING_IPC"
BUFFER_SIZE = 4096


class NamedPipeIPCServer:
    def __init__(self, orchestrator_callback=None):
        self.orchestrator_callback = orchestrator_callback
        self._running = True

    def _create_pipe_instance(self):
        return win32pipe.CreateNamedPipe(
            PIPE_NAME,
            win32pipe.PIPE_ACCESS_DUPLEX | win32file.FILE_FLAG_OVERLAPPED,
            win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
            win32pipe.PIPE_UNLIMITED_INSTANCES,
            BUFFER_SIZE,
            BUFFER_SIZE,
            50,  # Default wait 50ms
            None
        )

    async def run(self):
        loop = asyncio.get_running_loop()
        logger.info(f"Named Pipe IPC Server listening on {PIPE_NAME}...")

        while self._running:
            pipe_handle = self._create_pipe_instance()
            try:
                # Wait for MQL5 client connection without blocking the event loop
                await loop.run_in_executor(None, win32pipe.ConnectNamedPipe, pipe_handle, None)
                _, raw_bytes = await loop.run_in_executor(
                    None, win32file.ReadFile, pipe_handle, BUFFER_SIZE
                )
                
                try:
                    payload = json.loads(raw_bytes.decode("utf-8"))
                except Exception as e:
                    logger.error(f"Failed to decode IPC JSON: {e}")
                    payload = {}

                # Evaluate trade proposal (< 1ms via Redis cache / risk checks)
                if self.orchestrator_callback:
                    decision = await self.orchestrator_callback(payload)
                else:
                    decision = {"approved": True, "risk_pct": 1.0, "sl_mult": 1.2, "tp_mult": 1.25}

                # Compact pipe-delimited format for ultra-fast parsing in MQL5
                # Format: "approved|kelly_risk_pct|sl_atr_mult|tp_atr_mult"
                resp_str = (
                    f"{1 if decision.get('approved', False) else 0}|"
                    f"{decision.get('risk_pct', 1.0):.4f}|"
                    f"{decision.get('sl_mult', 1.2):.4f}|"
                    f"{decision.get('tp_mult', 1.25):.4f}"
                )

                await loop.run_in_executor(
                    None, win32file.WriteFile, pipe_handle, resp_str.encode("utf-8")
                )
                await loop.run_in_executor(None, win32file.FlushFileBuffers, pipe_handle)

            except pywintypes.error as e:
                # Normal disconnect from client
                pass
            except Exception as e:
                logger.error(f"IPC Pipe Server exception: {e}")
            finally:
                try:
                    win32pipe.DisconnectNamedPipe(pipe_handle)
                    win32file.CloseHandle(pipe_handle)
                except Exception:
                    pass

    # Alias for run loop
    handle_client_loop = run

    def stop(self):
        self._running = False
