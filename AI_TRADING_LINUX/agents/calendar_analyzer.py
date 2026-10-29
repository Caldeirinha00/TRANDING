import logging
import asyncio
from typing import Dict, Any, List, Set
import aiohttp
from datetime import datetime, timezone
from redis.asyncio import Redis

from core.config import CONFIG

logger = logging.getLogger(__name__)

# URL pública e gratuita do calendário econômico semanal (ForexFactory / Fair Economy)
CALENDAR_API_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Moedas principais monitoradas
_MAJOR_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD"}

# Mapeamento de índices e commodities para moeda base de impacto macro
_ASSET_CURRENCY_MAP = {
    "US500": {"USD"},
    "SPX": {"USD"},
    "US30": {"USD"},
    "NAS100": {"USD"},
    "GER40": {"EUR"},
    "UK100": {"GBP"},
    "JPN225": {"JPY"},
    "AUS200": {"AUD"},
    "FRA40": {"EUR"},
    "XAU": {"USD"},
    "GOLD": {"USD"},
    "XAG": {"USD"},
    "SILVER": {"USD"},
    "BRENT": {"USD"},
    "CRUDE": {"USD"},
    "WTI": {"USD"},
}


def _extract_currencies_from_symbol(symbol: str) -> Set[str]:
    """Identify relevant macroeconomic currencies for a given trading symbol."""
    s = symbol.upper().replace(".", "").replace("-", "").replace("_", "")
    currencies: Set[str] = set()

    for prefix, currs in _ASSET_CURRENCY_MAP.items():
        if prefix in s:
            currencies.update(currs)

    for curr in _MAJOR_CURRENCIES:
        if curr in s:
            currencies.add(curr)

    # Standard Forex fallback (e.g., EURUSD -> EUR, USD)
    if not currencies and len(s) >= 6:
        base, quote = s[:3], s[3:6]
        if base in _MAJOR_CURRENCIES:
            currencies.add(base)
        if quote in _MAJOR_CURRENCIES:
            currencies.add(quote)

    if not currencies:
        currencies.add("USD")

    return currencies


class CalendarAnalyzer:
    """
    Fetches real-time economic calendar events and enforces trading vetoes
    around high-impact macroeconomic releases (FOMC, CPI, NFP, GDP, etc.).
    """

    def __init__(self):
        self.redis_url = CONFIG.redis_url
        self.cache_ttl = 900  # 15 mins cache in Redis
        self._memory_cache: List[Dict[str, Any]] = []
        self._last_fetch_time: float = 0.0
        self._fetch_lock = asyncio.Lock()

    async def _fetch_calendar_events(self) -> List[Dict[str, Any]]:
        """Fetch economic events with memory caching to protect rate limits."""
        now_ts = asyncio.get_event_loop().time()
        if self._memory_cache and (now_ts - self._last_fetch_time < 300):
            return self._memory_cache

        async with self._fetch_lock:
            if self._memory_cache and (asyncio.get_event_loop().time() - self._last_fetch_time < 300):
                return self._memory_cache

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(CALENDAR_API_URL, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            if isinstance(data, list):
                                self._memory_cache = data
                                self._last_fetch_time = asyncio.get_event_loop().time()
                                logger.info(f"Economic calendar refreshed ({len(data)} events this week).")
                                return self._memory_cache
                        logger.warning(f"Calendar API responded with HTTP {resp.status}")
            except Exception as e:
                logger.error(f"Failed to fetch economic calendar: {e}")

            # Return stale cache if available upon failure
            return self._memory_cache

    async def get_upcoming_events(self, symbol: str) -> Dict[str, Any]:
        """
        Evaluates whether a high-impact event is imminent or recently occurred for `symbol`.
        Returns {'veto': bool, 'events': str}.
        """
        cache_key = f"calendar:{symbol}"

        # 1. Check Redis cache first
        try:
            async with Redis.from_url(self.redis_url, decode_responses=True) as redis:
                cached_status = await redis.get(cache_key)
                if cached_status:
                    is_veto = cached_status.startswith("veto")
                    reason = cached_status.split(":", 1)[1] if ":" in cached_status else "Cached status"
                    return {"veto": is_veto, "events": reason}
        except Exception as e:
            logger.debug(f"Redis calendar cache read failed: {e}")

        # 2. Fetch fresh events
        events = await self._fetch_calendar_events()
        if not events:
            return {"veto": False, "events": "No active calendar events found"}

        currencies = _extract_currencies_from_symbol(symbol)
        now = datetime.now(timezone.utc)
        veto = False
        upcoming_msgs = []

        # High-impact keywords that trigger broader safety windows (60 mins before/after)
        major_release_keywords = ["FOMC", "NFP", "CPI", "FED", "NON-FARM", "INTEREST RATE", "RATE DECISION", "GDP", "POWELL", "LAGARDE"]

        for e in events:
            impact = e.get("impact", "")
            country = e.get("country", "")

            # Match high-impact events for the symbol's currencies (or global events)
            if impact.lower() == "high" and (country in currencies or country == "All"):
                date_str = e.get("date")
                title = str(e.get("title", "")).upper()
                if not date_str:
                    continue

                try:
                    event_time = datetime.fromisoformat(date_str).astimezone(timezone.utc)
                    diff_minutes = (event_time - now).total_seconds() / 60.0

                    # Check window
                    is_critical = any(kw in title for kw in major_release_keywords)
                    pre_window = 60 if is_critical else 30
                    post_window = 30 if is_critical else 15

                    # If within [-post_window, +pre_window] minutes of release
                    if -post_window <= diff_minutes <= pre_window:
                        veto = True
                        time_display = event_time.strftime("%H:%M UTC")
                        if diff_minutes > 0:
                            upcoming_msgs.append(f"{title} ({country}) in {int(diff_minutes)}m [{time_display}]")
                        else:
                            upcoming_msgs.append(f"{title} ({country}) released {abs(int(diff_minutes))}m ago")
                except Exception as parse_err:
                    logger.debug(f"Error parsing calendar event date '{date_str}': {parse_err}")

        events_summary = " | ".join(upcoming_msgs) if upcoming_msgs else "No high-impact events within safety window"

        # 3. Store result in Redis
        try:
            status_val = f"veto:{events_summary}" if veto else f"safe:{events_summary}"
            async with Redis.from_url(self.redis_url, decode_responses=True) as redis:
                await redis.set(cache_key, status_val, ex=self.cache_ttl)
        except Exception as e:
            logger.debug(f"Redis calendar cache write failed: {e}")

        return {"veto": veto, "events": events_summary}


calendar_analyzer = CalendarAnalyzer()
