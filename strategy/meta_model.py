from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from backtest.validation import (
    PurgedEmbargoedWalkForwardSplit,
    compute_brier_skill_score,
    compute_sample_uniqueness_weights,
)
from strategy.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class MetaModelHyperParams:
    """
    Regularized shallow GBDT hyperparameters for small-sample FX Meta-Labeling.
    Strong tree depth/leaf constraints + L1/L2 penalties mitigate noise fitting.
    """

    n_estimators: int = 120
    learning_rate: float = 0.03
    max_depth: int = 3
    num_leaves: int = 7
    min_child_samples: int = 40
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 5.0
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    calibration_cv_splits: int = 3
    random_state: int = 42
    n_jobs: int = 1


@dataclass(frozen=True)
class MetaPredictionOutput:
    """Calibrated Layer 2 Meta-Labeling inference outputs."""

    meta_prob: np.ndarray
    expected_net_r: np.ndarray


@dataclass(frozen=True)
class WalkForwardMetaResult:
    """Aggregate Out-of-Sample Purged Walk-Forward Meta-Labeling outputs and diagnostics."""

    oos_predictions_df: pd.DataFrame
    meta_decisions_by_symbol: Dict[str, pd.DataFrame]
    fold_summaries: List[Dict[str, object]]
    aggregate_feature_importances: Dict[str, float]


class CalibratedLightGBMMetaModel:
    """
    Secondary Meta-Labeling classifier predicting P(realized_net_R > 0 | Layer-1 signal, regime)
    with in-fold probability calibration and causal net-payoff expectancy estimation.
    """

    def __init__(
        self,
        params: Optional[MetaModelHyperParams] = None,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.params = params or MetaModelHyperParams()
        self.feature_columns = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
        self.model: Optional[CalibratedClassifierCV | lgb.LGBMClassifier] = None
        self.fallback_base_rate: Optional[float] = None
        self.train_avg_win_r: float = 1.0
        self.train_avg_loss_r: float = -1.0
        self.is_fitted: bool = False

    def _to_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X[self.feature_columns].to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_base_estimator(self, effective_min_child_samples: int) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=self.params.n_estimators,
            learning_rate=self.params.learning_rate,
            max_depth=self.params.max_depth,
            num_leaves=self.params.num_leaves,
            min_child_samples=effective_mcs_value(effective_min_child_samples),
            subsample=self.params.subsample,
            subsample_freq=self.params.subsample_freq,
            colsample_bytree=self.params.colsample_bytree,
            reg_alpha=self.params.reg_alpha,
            reg_lambda=self.params.reg_lambda,
            random_state=self.params.random_state,
            n_jobs=self.params.n_jobs,
            verbose=-1,
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[int] | np.ndarray,
        realized_r: Sequence[float] | np.ndarray,
        sample_weight: Optional[Sequence[float] | np.ndarray] = None,
    ) -> "CalibratedLightGBMMetaModel":
        X_arr = self._to_matrix(X)
        y_arr = np.asarray(y, dtype=int)
        r_arr = np.asarray(realized_r, dtype=float)
        if sample_weight is None:
            w_arr = np.ones(len(y_arr), dtype=float)
        else:
            w_arr = np.asarray(sample_weight, dtype=float)

        win_mask = y_arr == 1
        loss_mask = y_arr == 0
        self.train_avg_win_r = (
            float(np.average(r_arr[win_mask], weights=w_arr[win_mask]))
            if np.any(win_mask) and np.sum(w_arr[win_mask]) > 0
            else 1.0
        )
        self.train_avg_loss_r = (
            float(np.average(r_arr[loss_mask], weights=w_arr[loss_mask]))
            if np.any(loss_mask) and np.sum(w_arr[loss_mask]) > 0
            else -1.0
        )

        self.fallback_base_rate = None
        self.model = None

        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2 or len(y_arr) < 15:
            base = float(np.average(y_arr, weights=w_arr)) if np.sum(w_arr) > 0 else 0.5
            self.fallback_base_rate = float(np.clip(base, 0.01, 0.99))
            self.is_fitted = True
            return self

        effective_mcs = min(self.params.min_child_samples, max(2, len(y_arr) // 5))
        chosen_cv: Optional[KFold] = None
        for n_splits in range(self.params.calibration_cv_splits, 1, -1):
            if len(y_arr) < n_splits * 6:
                continue
            kf = KFold(n_splits=n_splits, shuffle=False)
            valid_splits = all(
                len(np.unique(y_arr[tr])) == 2 and len(np.unique(y_arr[val])) == 2
                for tr, val in kf.split(X_arr)
            )
            if valid_splits:
                chosen_cv = kf
                break

        base_clf = self._build_base_estimator(effective_mcs)
        if chosen_cv is not None:
            cal_clf = CalibratedClassifierCV(
                estimator=base_clf,
                method=self.params.calibration_method,
                cv=chosen_cv,
            )
            cal_clf.fit(X_arr, y_arr, sample_weight=w_arr)
            self.model = cal_clf
        else:
            base_clf.fit(X_arr, y_arr, sample_weight=w_arr)
            self.model = base_clf

        self.is_fitted = True
        return self

    def predict_meta_probabilities(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("CalibratedLightGBMMetaModel must be fitted before prediction.")
        X_arr = self._to_matrix(X)
        if self.fallback_base_rate is not None or self.model is None:
            return np.full(len(X_arr), float(self.fallback_base_rate or 0.5), dtype=float)
        probs = self.model.predict_proba(X_arr)[:, 1]
        return np.clip(probs.astype(float), 0.0, 1.0)

    def predict_expected_net_r(self, meta_probs: Sequence[float] | np.ndarray) -> np.ndarray:
        p = np.asarray(meta_probs, dtype=float)
        return p * self.train_avg_win_r + (1.0 - p) * self.train_avg_loss_r

    def predict_decisions(self, X: pd.DataFrame | np.ndarray) -> MetaPredictionOutput:
        probs = self.predict_meta_probabilities(X)
        exp_r = self.predict_expected_net_r(probs)
        return MetaPredictionOutput(meta_prob=probs, expected_net_r=exp_r)

    def extract_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted or self.model is None:
            return {col: 0.0 for col in self.feature_columns}

        raw_importances: List[np.ndarray] = []
        if isinstance(self.model, CalibratedClassifierCV):
            for cal_est in self.model.calibrated_classifiers_:
                inner = cal_est.estimator
                if hasattr(inner, "feature_importances_"):
                    raw_importances.append(np.asarray(inner.feature_importances_, dtype=float))
        elif hasattr(self.model, "feature_importances_"):
            raw_importances.append(np.asarray(self.model.feature_importances_, dtype=float))

        if not raw_importances:
            return {col: 0.0 for col in self.feature_columns}

        mean_imp = np.mean(raw_importances, axis=0)
        total = float(np.sum(mean_imp))
        if total > 0.0:
            mean_imp = mean_imp / total
        return {col: float(mean_imp[i]) for i, col in enumerate(self.feature_columns)}


def effective_mcs_value(val: int) -> int:
    return max(2, int(val))


def run_purged_walk_forward_meta_labeling(
    events_df: pd.DataFrame,
    splitter: Optional[PurgedEmbargoedWalkForwardSplit] = None,
    params: Optional[MetaModelHyperParams] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> WalkForwardMetaResult:
    """
    Run strict chronological Purged & Embargoed Walk-Forward Meta-Labeling:
      - Computes sample uniqueness weights strictly inside each training fold
      - Restricts training events to resolved, risk/swap-eligible candidates when flagged
      - Emits strictly Out-of-Sample calibrated probabilities and expected net R per symbol
    """
    if events_df.empty:
        raise ValueError("Cannot run Walk-Forward Meta-Labeling on empty events_df.")

    df = events_df.sort_values(["signal_time", "symbol"], ascending=[True, True]).reset_index(drop=True)
    wf_splitter = splitter or PurgedEmbargoedWalkForwardSplit()
    cols = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)

    has_elig = "is_risk_swap_eligible" in df.columns
    has_resolved = "is_resolved_label" in df.columns

    folds = wf_splitter.split(df["signal_time"], df["exit_time"])
    if not folds:
        raise ValueError("PurgedEmbargoedWalkForwardSplit produced 0 valid folds.")

    oos_chunks: List[pd.DataFrame] = []
    fold_summaries: List[Dict[str, object]] = []
    imp_accum = np.zeros(len(cols), dtype=float)
    imp_folds = 0

    for fold in folds:
        raw_train_df = df.iloc[fold.train_indices].copy()
        if has_resolved:
            raw_train_df = raw_train_df[raw_train_df["is_resolved_label"]]
        if has_elig:
            elig_df = raw_train_df[raw_train_df["is_risk_swap_eligible"]]
            if len(elig_df) >= 30:
                raw_train_df = elig_df

        test_df = df.iloc[fold.test_indices].copy()
        if raw_train_df.empty or test_df.empty:
            continue

        train_weights = compute_sample_uniqueness_weights(
            start_times=raw_train_df["signal_time"],
            end_times=raw_train_df["exit_time"],
        )

        meta_model = CalibratedLightGBMMetaModel(params=params, feature_columns=cols)
        meta_model.fit(
            X=raw_train_df[cols],
            y=raw_train_df["meta_label"].to_numpy(dtype=int),
            realized_r=raw_train_df["realized_r"].to_numpy(dtype=float),
            sample_weight=train_weights,
        )

        pred = meta_model.predict_decisions(test_df[cols])
        test_df["fold_id"] = fold.fold_id
        test_df["meta_prob"] = pred.meta_prob
        test_df["expected_net_r"] = pred.expected_net_r
        test_df["train_avg_win_r"] = meta_model.train_avg_win_r
        test_df["train_avg_loss_r"] = meta_model.train_avg_loss_r
        oos_chunks.append(test_df)

        # Evaluate Brier Skill Score strictly on resolved, risk/swap-eligible test subset
        eval_mask = np.ones(len(test_df), dtype=bool)
        if has_resolved:
            eval_mask &= test_df["is_resolved_label"].to_numpy(dtype=bool)
        if has_elig:
            eval_mask &= test_df["is_risk_swap_eligible"].to_numpy(dtype=bool)

        eval_df = test_df.iloc[np.flatnonzero(eval_mask)] if np.any(eval_mask) else test_df
        bss_stats = compute_brier_skill_score(
            y_true=eval_df["meta_label"].to_numpy(dtype=int),
            y_prob=eval_df["meta_prob"].to_numpy(dtype=float),
        )

        f_imp = meta_model.extract_feature_importances()
        imp_accum += np.array([f_imp[c] for c in cols], dtype=float)
        imp_folds += 1

        fold_summaries.append(
            {
                "fold_id": fold.fold_id,
                "test_start": fold.test_start.strftime("%Y-%m-%d"),
                "test_end": fold.test_end.strftime("%Y-%m-%d"),
                "n_train": int(len(raw_train_df)),
                "n_test_all": int(len(test_df)),
                "n_test_eval": int(len(eval_df)),
                "purged": fold.purged_count,
                "embargoed": fold.embargoed_count,
                "train_win_rate": round(float(raw_train_df["meta_label"].mean()), 4),
                "test_win_rate": round(float(eval_df["meta_label"].mean()), 4),
                "mean_oos_prob": round(float(eval_df["meta_prob"].mean()), 4),
                "brier_skill_score": round(bss_stats["brier_skill_score"], 4),
            }
        )

    if not oos_chunks:
        raise ValueError("No Out-of-Sample predictions were produced across Walk-Forward folds.")

    oos_df = pd.concat(oos_chunks, axis=0, ignore_index=True)
    oos_df = oos_df.sort_values(["signal_time", "symbol"], ascending=[True, True]).reset_index(drop=True)

    meta_decisions_by_symbol: Dict[str, pd.DataFrame] = {}
    for sym, grp in oos_df.groupby("symbol", sort=True):
        sym_df = (
            grp.set_index("signal_time")[["meta_prob", "expected_net_r", "fold_id"]]
            .sort_index(ascending=True)
        )
        sym_df = sym_df[~sym_df.index.duplicated(keep="last")]
        meta_decisions_by_symbol[str(sym)] = sym_df

    agg_imp = {
        col: float(imp_accum[i] / max(1, imp_folds))
        for i, col in enumerate(cols)
    }

    return WalkForwardMetaResult(
        oos_predictions_df=oos_df,
        meta_decisions_by_symbol=meta_decisions_by_symbol,
        fold_summaries=fold_summaries,
        aggregate_feature_importances=agg_imp,
    )
