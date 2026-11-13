from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.metrics import brier_score_loss

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class WalkForwardFold:
    """Single Purged & Embargoed Walk-Forward train/test fold."""

    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purged_count: int
    embargoed_count: int


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio statistics."""

    n_obs: int
    n_trials_k: int
    periods_per_year: int
    observed_sr_per_period: float
    observed_sr_annualized: float
    sr_threshold_annualized: float
    skewness: float
    kurtosis_pearson: float
    psr_zero: float
    dsr: float


def compute_sample_uniqueness_weights(
    start_times: Sequence[pd.Timestamp] | pd.Series,
    end_times: Sequence[pd.Timestamp] | pd.Series,
) -> np.ndarray:
    """
    Vectorized O(N log N) sweep-line implementation of Lopez de Prado's Average Sample
    Uniqueness weighting (Ch. 4). Overlapping multi-asset trade intervals receive
    down-weighted importance proportional to 1 / concurrency_t, normalized to mean 1.0.
    """
    n_events = len(start_times)
    if n_events == 0:
        return np.array([], dtype=float)
    if n_events == 1:
        return np.array([1.0], dtype=float)

    starts_ns = pd.to_datetime(list(start_times), utc=True).astype("int64").to_numpy()
    ends_ns = pd.to_datetime(list(end_times), utc=True).astype("int64").to_numpy()
    # Guard zero-duration intervals by expanding minimally by 1ns
    eff_ends_ns = np.where(ends_ns <= starts_ns, starts_ns + 1, ends_ns)

    grid = np.unique(np.concatenate([starts_ns, eff_ends_ns]))
    grid.sort()
    n_intervals = len(grid) - 1
    if n_intervals <= 0:
        return np.ones(n_events, dtype=float)

    start_pos = np.searchsorted(grid, starts_ns)
    end_pos = np.searchsorted(grid, eff_ends_ns)

    diff = np.zeros(len(grid), dtype=np.int32)
    np.add.at(diff, start_pos, 1)
    np.add.at(diff, end_pos, -1)
    concurrency = np.cumsum(diff[:-1])

    durations = np.diff(grid).astype(np.float64)
    inv_conc_dur = np.where(concurrency > 0, durations / np.maximum(concurrency, 1), 0.0)

    pref_weighted = np.concatenate(([0.0], np.cumsum(inv_conc_dur)))
    pref_dur = np.concatenate(([0.0], np.cumsum(durations)))

    event_dur = pref_dur[end_pos] - pref_dur[start_pos]
    event_uniq_num = pref_weighted[end_pos] - pref_weighted[start_pos]
    raw_weights = np.where(event_dur > 0.0, event_uniq_num / event_dur, 1.0)

    mean_w = float(np.mean(raw_weights))
    if mean_w > 0.0:
        return raw_weights / mean_w
    return np.ones(n_events, dtype=float)


class PurgedEmbargoedWalkForwardSplit:
    """
    Chronological Walk-Forward generator with strict barrier-interval purging
    (removing any training sample whose `exit_time >= test_start - embargo_delta`)
    and optional embargo buffer against serial residual autocorrelation.
    """

    def __init__(
        self,
        initial_train_months: int = 48,
        test_step_months: int = 6,
        expanding: bool = True,
        rolling_window_months: int = 36,
        embargo_hours: int = 24,
        min_train_samples: int = 120,
        anchor_calendar_month_start: bool = True,
    ) -> None:
        self.initial_train_months = initial_train_months
        self.test_step_months = test_step_months
        self.expanding = expanding
        self.rolling_window_months = rolling_window_months
        self.embargo_hours = embargo_hours
        self.min_train_samples = min_train_samples
        self.anchor_calendar_month_start = anchor_calendar_month_start

    def split(
        self,
        signal_times: pd.Series | pd.DatetimeIndex,
        exit_times: pd.Series | pd.DatetimeIndex,
    ) -> List[WalkForwardFold]:
        sig_s = pd.Series(pd.to_datetime(signal_times, utc=True)).reset_index(drop=True)
        ext_s = pd.Series(pd.to_datetime(exit_times, utc=True)).reset_index(drop=True)
        if len(sig_s) != len(ext_s):
            raise ValueError("signal_times and exit_times must have identical lengths.")
        if len(sig_s) == 0:
            return []

        min_sig = sig_s.min()
        max_sig = sig_s.max()
        if self.anchor_calendar_month_start:
            base_anchor = pd.Timestamp(year=min_sig.year, month=min_sig.month, day=1, tz="UTC")
        else:
            base_anchor = min_sig

        embargo_td = pd.Timedelta(hours=max(0, int(self.embargo_hours)))
        cursor_start = base_anchor + pd.DateOffset(months=self.initial_train_months)
        folds: List[WalkForwardFold] = []
        fold_id = 0

        while cursor_start <= max_sig:
            cursor_end = cursor_start + pd.DateOffset(months=self.test_step_months)
            test_mask = (sig_s >= cursor_start) & (sig_s < cursor_end)
            test_idx = np.flatnonzero(test_mask.to_numpy())

            if len(test_idx) > 0:
                if self.expanding:
                    raw_train_mask = sig_s < cursor_start
                else:
                    roll_start = cursor_start - pd.DateOffset(months=self.rolling_window_months)
                    raw_train_mask = (sig_s >= roll_start) & (sig_s < cursor_start)

                purge_cutoff = cursor_start
                embargo_cutoff = cursor_start - embargo_td

                non_purged_mask = raw_train_mask & (ext_s < purge_cutoff)
                purged_count = int((raw_train_mask & ~non_purged_mask).sum())

                final_train_mask = non_purged_mask & (ext_s <= embargo_cutoff)
                embargoed_count = int((non_purged_mask & ~final_train_mask).sum())
                train_idx = np.flatnonzero(final_train_mask.to_numpy())

                if len(train_idx) >= self.min_train_samples:
                    folds.append(
                        WalkForwardFold(
                            fold_id=fold_id,
                            train_indices=train_idx,
                            test_indices=test_idx,
                            train_start=sig_s.iloc[train_idx].min(),
                            train_end=sig_s.iloc[train_idx].max(),
                            test_start=sig_s.iloc[test_idx].min(),
                            test_end=sig_s.iloc[test_idx].max(),
                            purged_count=purged_count,
                            embargoed_count=embargoed_count,
                        )
                    )
                    fold_id += 1

            cursor_start = cursor_end

        return folds


def compute_deflated_sharpe_ratio(
    returns: Sequence[float] | np.ndarray | pd.Series,
    num_trials_k: int = 1,
    periods_per_year: int = 252,
    trials_sr_variance: Optional[float] = None,
) -> DeflatedSharpeResult:
    """
    Compute Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR),
    correcting for non-normality (skewness & Pearson kurtosis) and multiple testing K.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_obs = len(arr)
    k = max(1, int(num_trials_k))

    if n_obs < 3 or float(np.std(arr, ddof=1)) <= 1e-12:
        return DeflatedSharpeResult(
            n_obs=n_obs,
            n_trials_k=k,
            periods_per_year=periods_per_year,
            observed_sr_per_period=0.0,
            observed_sr_annualized=0.0,
            sr_threshold_annualized=0.0,
            skewness=0.0,
            kurtosis_pearson=3.0,
            psr_zero=0.5,
            dsr=0.5,
        )

    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr, ddof=1))
    sr = mean_r / std_r
    sr_ann = sr * np.sqrt(periods_per_year)

    gam3 = float(skew(arr, bias=False)) if n_obs >= 4 else 0.0
    gam4 = float(kurtosis(arr, fisher=False, bias=False)) if n_obs >= 4 else 3.0
    if not np.isfinite(gam3):
        gam3 = 0.0
    if not np.isfinite(gam4) or gam4 < 1.0:
        gam4 = 3.0

    denom_sq = max(1e-12, 1.0 - gam3 * sr + ((gam4 - 1.0) / 4.0) * (sr ** 2))
    sr_se = np.sqrt(denom_sq / max(1, n_obs - 1))

    if k <= 1:
        sr_star = 0.0
    else:
        sr_var = (1.0 / n_obs) if trials_sr_variance is None else max(float(trials_sr_variance), 1e-12)
        z_max = (1.0 - EULER_MASCHERONI) * float(norm.ppf(1.0 - 1.0 / k)) + EULER_MASCHERONI * float(
            norm.ppf(1.0 - 1.0 / (k * np.e))
        )
        sr_star = float(np.sqrt(sr_var)) * z_max

    psr_zero = float(norm.cdf(sr / sr_se))
    dsr = float(norm.cdf((sr - sr_star) / sr_se))

    return DeflatedSharpeResult(
        n_obs=n_obs,
        n_trials_k=k,
        periods_per_year=periods_per_year,
        observed_sr_per_period=float(sr),
        observed_sr_annualized=float(sr_ann),
        sr_threshold_annualized=float(sr_star * np.sqrt(periods_per_year)),
        skewness=gam3,
        kurtosis_pearson=gam4,
        psr_zero=psr_zero,
        dsr=dsr,
    )


def compute_brier_skill_score(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    sample_weight: Optional[Sequence[float] | np.ndarray] = None,
) -> Dict[str, float]:
    """Compute Brier Score, climatological reference Brier Score, and Brier Skill Score (BSS)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    base_rate = float(np.average(yt, weights=sw)) if sw is not None else float(np.mean(yt))
    bs_model = float(brier_score_loss(yt, yp, sample_weight=sw))
    ref_probs = np.full_like(yt, base_rate, dtype=float)
    bs_ref = float(brier_score_loss(yt, ref_probs, sample_weight=sw))

    bss = (1.0 - (bs_model / bs_ref)) if bs_ref > 1e-12 else 0.0
    return {
        "base_rate": base_rate,
        "brier_model": bs_model,
        "brier_baseline": bs_ref,
        "brier_skill_score": float(bss),
    }
