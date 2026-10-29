import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CONFIG
from agents.orchestrator import orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "ipc_watcher.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("IPC_Watcher")

IPC_BASE = Path(r"C:\Users\mob70\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AI_TRADING_IPC")
REQ_DIR = IPC_BASE / "requests"
RES_DIR = IPC_BASE / "responses"
AUDIT_DIR = IPC_BASE / "audit"

FORBIDDEN_GDPR_KEYS = {"balance", "equity", "lot", "volume", "ticket", "account", "login", "password"}

class IPCWatcher:
    def __init__(self):
        self.req_dir = REQ_DIR
        self.res_dir = RES_DIR
        self.audit_dir = AUDIT_DIR
        self.decision_log = self.audit_dir / "LLM_Decision_Log.jsonl"
        self._ensure_dirs()
        self.startup_reconciliation()

    def _ensure_dirs(self):
        self.req_dir.mkdir(parents=True, exist_ok=True)
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def startup_reconciliation(self):
        """Purges stale orphaned files from prior crashes/restarts."""
        now = time.time()
        purged = 0
        for directory in [self.req_dir, self.res_dir]:
            for f in directory.glob("*.*"):
                # Purge files older than 30 seconds on startup
                if now - f.stat().st_mtime > 30:
                    try:
                        f.unlink()
                        purged += 1
                    except Exception as e:
                        logger.warning(f"Failed to purge orphan {f.name}: {e}")
        if purged > 0:
            logger.info(f"[Startup Reconciliation] Cleaned {purged} stale IPC files.")
        else:
            logger.info("[Startup Reconciliation] Clean state. No stale files detected.")

    def sanitize_gdpr(self, payload: dict) -> dict:
        """Strict GDPR compliance: removes any account/financial identification."""
        sanitized = {}
        for k, v in payload.items():
            if k.lower() in FORBIDDEN_GDPR_KEYS:
                logger.warning(f"[GDPR Alert] Stripped forbidden account key '{k}' from payload.")
                continue
            sanitized[k] = v
        return sanitized

    async def process_request(self, req_file: Path):
        try:
            # Read JSON payload
            with open(req_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.error(f"Failed reading request {req_file.name}: {e}")
            return

        payload = self.sanitize_gdpr(raw)
        req_uuid = payload.get("uuid", req_file.stem.replace("req_", ""))
        symbol = payload.get("symbol", "EURUSD")
        direction = payload.get("direction", "BUY")

        logger.info(f"[Handshake Received] UUID={req_uuid} | Symbol={symbol} | Direction={direction}")

        # Evaluate LLM Veto via Orchestrator
        is_vetoed = False
        reason = "APPROVED_BY_AI"
        confidence = 85

        try:
            is_vetoed = await orchestrator._apply_llm_veto(symbol, {"direction": direction})
            if is_vetoed:
                reason = "VETOED_EXTREME_COUNTER_SENTIMENT"
                confidence = 90
        except Exception as e:
            logger.error(f"LLM Veto evaluation failed: {e}. Defaulting to Fail-Closed.")
            is_vetoed = True
            reason = f"FAIL_CLOSED_EXCEPTION_{type(e).__name__}"

        response_payload = {
            "uuid": req_uuid,
            "symbol": symbol,
            "direction": direction,
            "vetoed": is_vetoed,
            "reason": reason,
            "confidence": confidence,
            "model_hash": CONFIG.LLM_MODEL,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Write atomic response: .tmp -> .json
        tmp_res = self.res_dir / f"res_{req_uuid}.json.tmp"
        final_res = self.res_dir / f"res_{req_uuid}.json"

        try:
            with open(tmp_res, "w", encoding="utf-8") as f:
                json.dump(response_payload, f, indent=2)
            
            # Atomic rename on Windows
            if final_res.exists():
                final_res.unlink()
            tmp_res.rename(final_res)

            # Audit log
            with open(self.decision_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(response_payload) + "\n")

            logger.info(f"[Handshake Responded] UUID={req_uuid} | Vetoed={is_vetoed} | Reason={reason}")

        except Exception as e:
            logger.error(f"Failed to write atomic response for {req_uuid}: {e}")

        # Clean up processed request file
        try:
            req_file.unlink()
        except Exception:
            pass

    async def run(self):
        logger.info("=== AI_TRADING IPC Watcher Active (Listening for MT5 Signals) ===")
        while True:
            try:
                # Find complete .json requests (ignoring .tmp)
                for req_file in list(self.req_dir.glob("req_*.json")):
                    await self.process_request(req_file)
            except Exception as e:
                logger.error(f"Error in IPC poll loop: {e}")
            await asyncio.sleep(0.05)

if __name__ == "__main__":
    watcher = IPCWatcher()
    try:
        asyncio.run(watcher.run())
    except KeyboardInterrupt:
        logger.info("IPC Watcher stopped by user.")
