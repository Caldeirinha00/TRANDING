"""
Unified MetaTrader 5 Bridge for Linux and Windows.
Connects to the isolated MT5 Docker container (RPyC on port 18812).
"""
import logging
import os

logger = logging.getLogger(__name__)

mt5 = None

try:
    import MetaTrader5 as _mt5
    mt5 = _mt5
    logger.debug("Loaded native Windows MetaTrader5 module.")
except ImportError:
    try:
        from mt5linux import MetaTrader5 as _MetaTrader5Linux
        from core.config import CONFIG

        host = getattr(CONFIG, "MT5_HOST", os.getenv("MT5_HOST", "127.0.0.1"))
        port = int(getattr(CONFIG, "MT5_PORT", os.getenv("MT5_PORT", "18812")))

        mt5 = _MetaTrader5Linux(
            host=host,
            port=port,
            mt5_login=getattr(CONFIG, "MT5_LOGIN", None),
            mt5_password=getattr(CONFIG, "MT5_PASSWORD", None),
            mt5_server=getattr(CONFIG, "MT5_SERVER", None),
        )
        logger.info(f"Initialized mt5linux bridge client connecting to {host}:{port}")
    except ImportError:
        logger.warning("mt5linux not found. Please install: pip install mt5linux")
    except Exception as e:
        logger.error(f"Failed to initialize mt5linux bridge: {e}")
