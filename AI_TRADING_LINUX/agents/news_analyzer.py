import asyncio
import logging
import json
import aiohttp
from typing import Dict, Any, List
from datetime import datetime
import hashlib
import time
import re
try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from agents.llm.mistral_client import mistral_client
from core.config import CONFIG

logger = logging.getLogger(__name__)

# Cache to store the latest raw news hash and its corresponding LLM response per symbol
_news_hash_cache: Dict[str, Dict[str, Any]] = {}

# Time-based cache to prevent Yahoo Finance IP bans (10 minutes per ticker)
_yahoo_api_cache: Dict[str, Dict[str, Any]] = {}

# Common crypto base currencies
_CRYPTO_BASES = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "LINK", "MATIC"}
# Common forex quote currencies
_FOREX_QUOTES = {"USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"}
# Index symbols
_INDEX_NAMES = {"US500", "US30", "NAS100", "GER40", "UK100", "JPN225", "AUS200", "FRA40", "NETH25", "SPX"}
# Commodity keywords
_COMMODITY_KEYWORDS = {"XAU", "XAG", "GOLD", "SILVER", "CRUDE", "BRENT", "WTI", "XTI", "UKO", "SPOTCRUDE", "SPOTBRENT"}


def _detect_asset_class(symbol: str) -> str:
    """Detect asset class from symbol name. Returns 'crypto', 'forex', 'commodity', 'index', or 'stock'."""
    s = symbol.upper().replace(".", "").replace("-", "").replace("_", "")
    
    # Check index first (US500, NAS100, GER40 etc.)
    for idx in _INDEX_NAMES:
        if idx in s:
            return "index"
    
    # Check commodity (XAU, XAG, Crude, Brent etc.)
    for kw in _COMMODITY_KEYWORDS:
        if kw in s:
            return "commodity"
    
    # Strip common broker suffixes like '.US-24', '.NAS'
    for suffix in ["US24", "NAS", "NYSE", "US"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            return "stock"
    # Check crypto
    for base in _CRYPTO_BASES:
        if s.startswith(base):
            return "crypto"
    # Check forex (6 chars, both halves are currencies)
    if len(s) == 6 and s[:3] in _FOREX_QUOTES and s[3:] in _FOREX_QUOTES:
        return "forex"
    # Default to stock
    return "stock"


def _get_yahoo_ticker(symbol: str, asset_class: str) -> str:
    """Convert MT5 symbol to Yahoo Finance ticker format."""
    s = symbol.upper()
    if asset_class == "crypto":
        # BTCUSD -> BTC-USD
        for base in _CRYPTO_BASES:
            if s.startswith(base):
                return f"{base}-USD"
        return f"{s[:3]}-USD"
    elif asset_class == "forex":
        return f"{s[:6]}=X"
    elif asset_class == "commodity":
        if "XAU" in s or "GOLD" in s:
            return "GC=F"
        elif "XAG" in s or "SILVER" in s:
            return "SI=F"
        elif "CRUDE" in s or "WTI" in s or "XTI" in s:
            return "CL=F"
        elif "BRENT" in s or "UKO" in s:
            return "BZ=F"
        return s
    elif asset_class == "index":
        if "US500" in s or "SPX" in s:
            return "^GSPC"
        elif "NAS100" in s:
            return "^IXIC"
        elif "US30" in s:
            return "^DJI"
        elif "GER40" in s:
            return "^GDAXI"
        return s
    else:
        # Stock: strip broker suffixes (e.g. .US-24, .NYSE) safely using regex
        clean = re.split(r'[\.\-]', s)[0]
        return clean


class NewsAnalyzer:
    """Fundamental analysis agent. Fetches macro calendar + news sentiment."""

    async def _fetch_fmp_calendar(self, session: aiohttp.ClientSession) -> str:
        # A API do calendário macro do FMP passou a ser paga. Retornamos vazio.
        return "No major macro events today (Calendar disabled)."

    async def _fetch_yahoo_news(self, session: aiohttp.ClientSession, symbol: str, asset_class: str) -> str:
        """Fetch latest news from Yahoo Finance RSS."""
        yahoo_ticker = _get_yahoo_ticker(symbol, asset_class)
        
        # 1. Rate Limiting / Caching to avoid IP Ban (5 minutes)
        now = time.time()
        if yahoo_ticker in _yahoo_api_cache:
            if now - _yahoo_api_cache[yahoo_ticker]["time"] < 300:
                return _yahoo_api_cache[yahoo_ticker]["data"]
        
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={yahoo_ticker}&region=US&lang=en-US"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        try:
            # 2. Timeout obrigatório (5s para não atrasar ciclo de 15s)
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    xml_content = await resp.text()
                    
                    # 3. Parsing seguro com try-except ET.ParseError
                    try:
                        root = ET.fromstring(xml_content)
                    except ET.ParseError:
                        logger.warning(f"Yahoo RSS retornou XML inválido para {yahoo_ticker}")
                        return f"Yahoo Finance returned invalid XML for {yahoo_ticker}."
                        
                    items = root.findall('.//item')
                    snippets = []
                    for item in items[:5]:
                        title = item.find('title')
                        desc = item.find('description')
                        t_text = title.text if title is not None else ""
                        d_text = desc.text if desc is not None else ""
                        if t_text:
                            snippets.append(f"- {t_text}: {d_text[:150]}")
                            
                    result = (f"Recent News for {yahoo_ticker}:\n" + "\n".join(snippets)) if snippets else f"No recent news for {yahoo_ticker}."
                    
                    # Update HTTP Cache
                    _yahoo_api_cache[yahoo_ticker] = {"time": now, "data": result}
                    return result
                    
                return f"Yahoo Finance returned status {resp.status}"
        except asyncio.TimeoutError:
             logger.warning(f"Yahoo RSS timeout para {yahoo_ticker}")
             return f"Yahoo Finance unavailable (Timeout)."
        except Exception as e:
            logger.error(f"Yahoo fetch error: {e}")
            return f"Yahoo Finance unavailable: {e}"

    async def analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Gathers macro calendar + news, sends to LLM for structured sentiment analysis.
        Uses async context manager for safe session handling.
        """
        asset_class = _detect_asset_class(symbol)
        asset_label = {"crypto": "cryptocurrency", "forex": "forex pair", "stock": "stock", "commodity": "commodity", "index": "market index"}[asset_class]
        logger.info(f"NewsAnalyzer gathering data for {symbol} (detected: {asset_label})...")

        async with aiohttp.ClientSession() as session:
            fmp_data, yahoo_data = await asyncio.gather(
                self._fetch_fmp_calendar(session),
                self._fetch_yahoo_news(session, symbol, asset_class),
                return_exceptions=True
            )
            # Handle exceptions from gather
            if isinstance(fmp_data, Exception):
                fmp_data = f"FMP error: {fmp_data}"
            if isinstance(yahoo_data, Exception):
                yahoo_data = f"Yahoo Finance error: {yahoo_data}"

        prompt = f"""Analyze the following real-time macroeconomic and news data for the {asset_label} {symbol}.

[MACROECONOMIC CALENDAR]
{fmp_data}

[LATEST NEWS]
{yahoo_data}

Based on this combined data, what is the fundamental sentiment for {symbol} right now?
Return ONLY a JSON object with this exact schema:
{{
    "symbol": "{symbol}",
    "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
    "confidence": 0-100,
    "reasoning_summary": "One sentence explaining why"
}}
"""
        
        # 1. Generate MD5 Hash of the raw text
        raw_text = f"{fmp_data}\n\n{yahoo_data}"
        text_hash = hashlib.md5(raw_text.encode('utf-8')).hexdigest()
        
        # 2. Check if we already processed this exact news
        cached_data = _news_hash_cache.get(symbol)
        if cached_data and cached_data["hash"] == text_hash:
            logger.info(f"News for {symbol} hasn't changed. Using cached sentiment: {cached_data['result'].get('sentiment')} (0 tokens used).")
            return cached_data["result"]
            
        # 3. If news changed (or FMP/Yahoo failed), call LLM
        # But wait, if FMP and Yahoo both failed, maybe we shouldn't call LLM?
        if "FMP error" in str(fmp_data) and "Yahoo Finance error" in str(yahoo_data):
            logger.warning(f"Both News APIs failed for {symbol}. Returning NEUTRAL to protect system.")
            return {
                "symbol": symbol,
                "sentiment": "NEUTRAL",
                "confidence": 0,
                "reasoning_summary": "News APIs unavailable."
            }

        result = await mistral_client.get_json_analysis(
            prompt=prompt,
            system_prompt="You are an elite quantitative macro analyst. Output ONLY valid JSON."
        )

        if result:
            logger.info(f"Sentiment for {symbol}: {result.get('sentiment')} ({result.get('confidence')}%)")
            # 4. Save to cache
            _news_hash_cache[symbol] = {"hash": text_hash, "result": result}
            return result
            
        return {
            "symbol": symbol,
            "sentiment": "NEUTRAL",
            "confidence": 0,
            "reasoning_summary": "Failed to get LLM analysis."
        }

news_analyzer = NewsAnalyzer()
