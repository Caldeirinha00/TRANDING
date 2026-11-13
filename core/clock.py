"""
MT5 Server H4 Clock & Session Synchronization.

All bar timestamps stored in QuestDB and returned by MetaTrader 5 encode the
broker server clock (Pepperstone EET UTC+2 / EEST UTC+3) directly as UTC
epoch timestamps. This module synchronizes wakeups to the broker's server H4
bar closes (00:00:02, 04:00:02, 08:00:02, 12:00:02, 16:00:02, 20:00:02 server
time) rather than assuming raw UTC boundaries.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Protocol

logger = logging.getLogger(__name__)

H4_INTERVAL_HOURS = 4
H4_OPEN_HOURS = (0, 4, 8, 12, 16, 20)


class ServerTimeProvider(Protocol):
    def get_server_time_utc(self) -> Optional[datetime]: ...


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class H4ServerClock:
    """
    Broker-synchronized H4 bar clock supporting deterministic backtest/unit
    testing as well as live MT5 server offset tracking across EET/EEST DST shifts.
    """

    def __init__(
        self,
        offset_seconds: float = 2.0,
        server_minus_utc_seconds: Optional[float] = None,
    ) -> None:
        self.offset_seconds = max(0.0, float(offset_seconds))
        self._server_minus_utc_seconds: Optional[float] = (
            float(server_minus_utc_seconds) if server_minus_utc_seconds is not None else None
        )

    @property
    def server_minus_utc_seconds(self) -> Optional[float]:
        return self._server_minus_utc_seconds

    def sync_offset_from_server_time(
        self,
        server_now: datetime,
        real_utc_now: Optional[datetime] = None,
    ) -> float:
        """
        Records the current broker-server minus real-UTC delta (rounded to nearest
        minute so minor sub-second network jitter does not drift scheduling).
        """
        srv = ensure_utc(server_now)
        utc_ref = ensure_utc(real_utc_now or datetime.now(timezone.utc))
        raw_delta = (srv - utc_ref).total_seconds()
        # Broker offsets are integer hour/half-hour steps (typically +7200 or +10800s).
        rounded_delta = round(raw_delta / 60.0) * 60.0
        self._server_minus_utc_seconds = float(rounded_delta)
        return self._server_minus_utc_seconds

    def estimate_server_now(
        self,
        connector: Optional[ServerTimeProvider] = None,
        real_utc_now: Optional[datetime] = None,
    ) -> datetime:
        """
        Returns the best known broker server timestamp (encoded as UTC tzinfo).
        Uses live connector server time when available and falls back to the last
        synchronized offset against real UTC.
        """
        utc_ref = ensure_utc(real_utc_now or datetime.now(timezone.utc))
        if connector is not None:
            live_srv = connector.get_server_time_utc()
            if live_srv is not None:
                live_srv_utc = ensure_utc(live_srv)
                self.sync_offset_from_server_time(live_srv_utc, real_utc_now=utc_ref)
                return live_srv_utc

        if self._server_minus_utc_seconds is not None:
            return utc_ref + timedelta(seconds=self._server_minus_utc_seconds)

        # Conservative default for Pepperstone EET (UTC+2) before first broker ping.
        return utc_ref + timedelta(hours=2)

    @staticmethod
    def current_h4_bar_open(server_now: datetime) -> datetime:
        srv = ensure_utc(server_now)
        h4_hour = (srv.hour // H4_INTERVAL_HOURS) * H4_INTERVAL_HOURS
        return srv.replace(hour=h4_hour, minute=0, second=0, microsecond=0)

    @classmethod
    def latest_closed_h4_bar_open(cls, server_now: datetime) -> datetime:
        return cls.current_h4_bar_open(server_now) - timedelta(hours=H4_INTERVAL_HOURS)

    @staticmethod
    def latest_closed_d1_bar_open(server_now: datetime) -> datetime:
        srv = ensure_utc(server_now)
        midnight = srv.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight - timedelta(days=1)

    @staticmethod
    def is_weekend_market_closed(server_now: datetime) -> bool:
        """
        Spot FX closes on Friday 24:00 (Saturday 00:00 server time) and reopens
        Sunday 24:00 (Monday 00:00 server time). We leave a 5-minute buffer around
        Saturday 00:00 so Friday's 20:00 H4 close cycle runs cleanly.
        """
        srv = ensure_utc(server_now)
        wd = srv.weekday()  # Mon=0 .. Fri=4, Sat=5, Sun=6
        if wd == 5:
            return (srv.hour > 0) or (srv.hour == 0 and srv.minute >= 5)
        if wd == 6:
            return (srv.hour < 23) or (srv.hour == 23 and srv.minute < 55)
        return False

    @staticmethod
    def is_trading_h4_bar_open(bar_open: datetime) -> bool:
        b = ensure_utc(bar_open)
        return b.weekday() < 5 and b.hour in H4_OPEN_HOURS

    def next_h4_close_wakeup_server(
        self,
        server_now: datetime,
        skip_weekend: bool = False,
    ) -> datetime:
        """
        Returns the next scheduled wakeup timestamp (in server clock encoded as UTC)
        at H4_close + self.offset_seconds.
        """
        srv = ensure_utc(server_now)
        cur_open = self.current_h4_bar_open(srv)
        # Check whether we are right at the boundary before T + offset_seconds.
        if (
            cur_open.weekday() < 5 or (cur_open.weekday() == 5 and cur_open.hour == 0)
        ):
            boundary_wakeup = cur_open + timedelta(seconds=self.offset_seconds)
            if srv < boundary_wakeup:
                if not skip_weekend or not self.is_weekend_market_closed(boundary_wakeup):
                    return boundary_wakeup

        candidate = cur_open + timedelta(hours=H4_INTERVAL_HOURS) + timedelta(seconds=self.offset_seconds)
        if not skip_weekend:
            return candidate

        while True:
            # A wakeup at Saturday 00:00:02 closes Friday's 20:00 H4 bar and IS valid.
            closed_bar_open = self.latest_closed_h4_bar_open(candidate)
            if self.is_trading_h4_bar_open(closed_bar_open):
                return candidate
            candidate += timedelta(hours=H4_INTERVAL_HOURS)

    def seconds_until_next_h4_wakeup(
        self,
        server_now: datetime,
        skip_weekend: bool = False,
    ) -> float:
        srv = ensure_utc(server_now)
        nxt = self.next_h4_close_wakeup_server(srv, skip_weekend=skip_weekend)
        return max(0.0, (nxt - srv).total_seconds())

    @classmethod
    def count_trading_h4_bars_between(
        cls,
        entry_bar_open: datetime,
        latest_closed_bar_open: datetime,
        inclusive_entry_bar: bool = True,
    ) -> int:
        """
        Counts completed trading H4 bars elapsed up to `latest_closed_bar_open`.
        With `inclusive_entry_bar=True` (default), the bar on whose open the
        position entered counts as bar #1 upon closing, matching `backtest/engine.py`.
        """
        start = cls.current_h4_bar_open(ensure_utc(entry_bar_open))
        end = cls.current_h4_bar_open(ensure_utc(latest_closed_bar_open))
        if end < start:
            return 0
        count = 0
        cur = start if inclusive_entry_bar else (start + timedelta(hours=H4_INTERVAL_HOURS))
        while cur <= end:
            if cls.is_trading_h4_bar_open(cur):
                count += 1
            cur += timedelta(hours=H4_INTERVAL_HOURS)
        return count

    async def wait_for_next_h4_close(
        self,
        connector: Optional[ServerTimeProvider] = None,
        skip_weekend: bool = True,
        max_sleep_chunk_seconds: float = 30.0,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> datetime:
        """
        Asynchronously sleeps in bounded chunks until the next H4 bar close + offset,
        re-synchronizing against live broker server time if the clock wakes up near DST.
        Returns the server timestamp of the closed H4 bar open `(T)` to process.
        """
        while True:
            if stop_check is not None and stop_check():
                return self.latest_closed_h4_bar_open(self.estimate_server_now(connector))
            srv_now = self.estimate_server_now(connector)
            remaining = self.seconds_until_next_h4_wakeup(srv_now, skip_weekend=skip_weekend)
            if remaining <= 0.05:
                # Re-read live server time right at boundary to confirm close passed.
                confirmed_now = self.estimate_server_now(connector)
                post_remaining = self.seconds_until_next_h4_wakeup(
                    confirmed_now, skip_weekend=skip_weekend
                )
                # Note: once confirmed_now >= T_close + offset, post_remaining jumps to ~14400s.
                if post_remaining > max(self.offset_seconds + 1.0, 10.0) or remaining <= 0.01:
                    return self.latest_closed_h4_bar_open(confirmed_now)
                sleep_for = min(post_remaining, 0.25)
            else:
                sleep_for = min(remaining, max_sleep_chunk_seconds)
            await asyncio.sleep(max(0.05, sleep_for))
