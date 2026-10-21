"""
模型比較實驗：LightGBM vs Logistic Regression vs Random Forest
================================================================

這是「重構之後」新增的實驗模組，不屬於原始 notebook 的內容。
目的：在完全相同的資料、特徵、時間切分之下，比較不同模型的表現，
回答「LightGBM 是不是真的比簡單模型好」這個原始版本沒有驗證過的問題。

公平比較的原則
--------------
1. 同一份特徵：所有模型都吃 LaggedFeatureBuilder 產生的同一組滯後特徵。
2. 同一個時間切分：time_based_split 只呼叫一次，所有模型共用同一組
   訓練/驗證日期，驗證集永遠在訓練集之後。
3. 驗證集不參與任何訓練決策：
   原本 DayTradeModel 用驗證集做 LightGBM 的 early stopping，等於讓驗證集
   參與了「選幾棵樹」的決策，對 LightGBM 稍微有利。比較實驗裡改成從
   訓練期間的「最後 10% 日期」切出內部 early-stopping 集，驗證集完全不碰。
4. 類別不平衡處理一致：LR / RF 用 class_weight="balanced"，
   LightGBM 用等價的 scale_pos_weight = 負樣本數 / 正樣本數。
5. 同時報告「門檻無關」的指標：AUC-ROC 與每日 Top-N 命中率
   （label 本身就是「每天 Top-N」，所以每天挑分數最高的 N 檔、
   看有幾檔真的是當沖股，是最貼近業務用途的指標）。

另外加入一個不需要訓練的 naive baseline：直接拿「昨天的當沖比例」排序。
任何模型如果贏不過它，代表模型其實沒有學到比「昨天熱門、今天也熱門」更多的東西。

此外也比較兩種三模型融合方式（Average / 時間序列版 Stacking），實作在 ensemble.py。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import LABEL_CONFIG, LGB_CONFIG, MODEL_CONFIG
from .data_interface import MarketDataSource
from .ensemble import AverageEnsemble, StackingEnsemble
from .feature_engineering import LaggedFeatureBuilder
from .model import time_based_split
from .pipeline import load_and_prepare

LABEL_COL = "是否為當沖股"
BASELINE_FEATURE = "當沖比例_lag1"


# ---------------------------------------------------------------------------
# 資料準備：只做一次，所有模型共用
# ---------------------------------------------------------------------------
@dataclass
class ComparisonData:
    X_train: np.ndarray
    y_train: np.ndarray
    train_dates: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    val_dates: np.ndarray
    feature_names: List[str]


def prepare_comparison_data(
    df: pd.DataFrame,
    n_lags: int = MODEL_CONFIG.n_lags,
    validation_ratio: float = MODEL_CONFIG.validation_ratio,
) -> ComparisonData:
    builder = LaggedFeatureBuilder(n_lags=n_lags)
    df_processed = builder.fit_transform(df, label_col=LABEL_COL)
    train_df, val_df = time_based_split(df_processed, validation_ratio)

    cols = builder.lagged_feature_cols
    return ComparisonData(
        X_train=train_df[cols].to_numpy(dtype=float),
        y_train=train_df[LABEL_COL].to_numpy(dtype=int),
        train_dates=train_df["日期"].to_numpy(),
        X_val=val_df[cols].to_numpy(dtype=float),
        y_val=val_df[LABEL_COL].to_numpy(dtype=int),
        val_dates=val_df["日期"].to_numpy(),
        feature_names=list(cols),
    )


# ---------------------------------------------------------------------------
# 候選模型：統一成 fit(X, y, dates) / predict_proba(X) 介面
# ---------------------------------------------------------------------------
class SklearnCandidate:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, dates, feature_names):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)[:, 1]


class LightGBMCandidate:
    """沿用 config.py 的 LightGBM 參數，但 early stopping 只用訓練期間內部切出的資料。"""

    def __init__(self, inner_validation_ratio: float = 0.1, random_state: int = 42):
        self.inner_validation_ratio = inner_validation_ratio
        self.random_state = random_state
        self.booster: Optional[lgb.Booster] = None

    def fit(self, X, y, dates, feature_names):
        unique_dates = np.sort(np.unique(dates))
        cut = unique_dates[int(len(unique_dates) * (1 - self.inner_validation_ratio))]
        fit_mask = dates < cut

        y_fit = y[fit_mask]
        n_pos = y_fit.sum()
        scale_pos_weight = (len(y_fit) - n_pos) / n_pos if n_pos > 0 else 1.0

        params = LGB_CONFIG.to_lgb_params(scale_pos_weight)
        params.update(seed=self.random_state, deterministic=True, force_row_wise=True)

        fit_set = lgb.Dataset(X[fit_mask], label=y_fit)
        stop_set = lgb.Dataset(X[~fit_mask], label=y[~fit_mask], reference=fit_set)
        self.booster = lgb.train(
            params,
            fit_set,
            num_boost_round=LGB_CONFIG.num_boost_round,
            valid_sets=[stop_set],
            callbacks=[lgb.early_stopping(LGB_CONFIG.early_stopping_rounds, verbose=False)],
        )
        return self

    def predict_proba(self, X):
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)


class LaggedRatioBaseline:
    """不訓練：分數 = 昨天的當沖比例。用來檢查模型有沒有學到比這更多的東西。"""

    is_rule_based = True

    def fit(self, X, y, dates, feature_names):
        self.col_idx = feature_names.index(BASELINE_FEATURE)
        return self

    def predict_proba(self, X):
        return X[:, self.col_idx]


def base_model_factories(random_state: int = 42) -> Dict[str, Callable[[], object]]:
    """三個要比較/融合的成員模型。"""
    return {
        "Logistic Regression": lambda: SklearnCandidate(
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state),
            )
        ),
        "Random Forest": lambda: SklearnCandidate(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=50,  # 對齊 LightGBM 的 min_data_in_leaf
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            )
        ),
        "LightGBM": lambda: LightGBMCandidate(random_state=random_state),
    }


def default_candidates(random_state: int = 42, include_ensembles: bool = True) -> Dict[str, Callable[[], object]]:
    """回傳 {模型名稱: 建構函式}，每次實驗重新建立，避免狀態殘留。"""
    members = base_model_factories(random_state)
    candidates: Dict[str, Callable[[], object]] = {
        "Baseline (當沖比例_lag1)": lambda: LaggedRatioBaseline(),
        **members,
    }
    if include_ensembles:
        candidates["Ensemble: Average"] = lambda: AverageEnsemble(members)
        candidates["Ensemble: Stacking"] = lambda: StackingEnsemble(members, random_state=random_state)
    return candidates


# ---------------------------------------------------------------------------
# 評估：同一套指標套在所有模型上
# ---------------------------------------------------------------------------
def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
    threshold: Optional[float] = MODEL_CONFIG.predict_threshold,
    top_n: int = LABEL_CONFIG.top_n_per_day,
) -> Dict[str, float]:
    """
    threshold=None 代表分數不是機率（例如 baseline），只計算門檻無關的指標。

    每日命中率：沿用 DayTradeModel.evaluate() 的定義（當天預測對的當沖股數 / 當天實際當沖股數）。
    每日 Top-N 命中率：每天取分數最高的 N 檔，其中真的是當沖股的比例。
    """
    frame = pd.DataFrame({"date": dates, "y": y_true, "score": scores})

    top = frame.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(top_n)
    daily_top_n = top.groupby("date")["y"].mean()

    result = {
        "AUC-ROC": roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan,
        f"Top-{top_n} Hit Rate": float(daily_top_n.mean()),
    }

    if threshold is None:
        result.update({k: np.nan for k in ("Accuracy", "Precision", "Recall", "F1", "Daily Hit Rate")})
        return result

    y_pred = (scores >= threshold).astype(int)
    frame["tp"] = (y_pred == 1) & (y_true == 1)
    daily = frame.groupby("date").agg(tp=("tp", "sum"), actual=("y", "sum"))
    daily_hit = (daily["tp"] / daily["actual"].replace(0, np.nan)).mean()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result.update(
        {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Daily Hit Rate": float(daily_hit),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        }
    )
    return result


def compare_models(
    data: ComparisonData,
    candidates: Optional[Dict[str, Callable[[], object]]] = None,
) -> pd.DataFrame:
    candidates = candidates or default_candidates()
    rows = []
    for name, factory in candidates.items():
        model = factory()
        start = time.perf_counter()
        model.fit(data.X_train, data.y_train, data.train_dates, data.feature_names)
        fit_seconds = time.perf_counter() - start

        scores = model.predict_proba(data.X_val)
        threshold = None if getattr(model, "is_rule_based", False) else MODEL_CONFIG.predict_threshold
        metrics = evaluate_scores(data.y_val, scores, data.val_dates, threshold=threshold)
        row = {"model": name, **metrics, "Fit Seconds": fit_seconds}
        if hasattr(model, "meta_weights"):
            row.update({f"meta_weight[{k}]": v for k, v in model.meta_weights.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def run_model_comparison(
    source_factory: Callable[[int], MarketDataSource],
    start_date: str,
    end_date: str,
    seeds: List[int],
    candidates: Optional[Dict[str, Callable[[], object]]] = None,
) -> pd.DataFrame:
    """
    對每一個 seed 重新產生一份 Mock 資料、重跑一次比較，回傳所有 seed 的明細。
    用多個 seed 是為了看出差異是否穩定，而不是某一份亂數剛好有利於某個模型。
    """
    all_runs = []
    for seed in seeds:
        df = load_and_prepare(source_factory(seed), start_date, end_date)
        data = prepare_comparison_data(df)
        result = compare_models(data, candidates)
        result.insert(0, "seed", seed)
        result["n_train"] = len(data.y_train)
        result["n_val"] = len(data.y_val)
        result["val_pos_ratio"] = data.y_val.mean()
        result["train_period"] = f"{data.train_dates.min()}~{data.train_dates.max()}"
        result["val_period"] = f"{data.val_dates.min()}~{data.val_dates.max()}"
        all_runs.append(result)
    return pd.concat(all_runs, ignore_index=True)


SUMMARY_METRICS = ["AUC-ROC", f"Top-{LABEL_CONFIG.top_n_per_day} Hit Rate", "Accuracy", "Precision", "Recall", "F1", "Daily Hit Rate"]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """把多個 seed 的結果整理成 mean ± std 的表格，保留原本的模型順序。"""
    order = list(dict.fromkeys(runs["model"]))
    grouped = runs.groupby("model", sort=False)[SUMMARY_METRICS]
    mean, std = grouped.mean().loc[order], grouped.std(ddof=0).loc[order]

    def fmt(m, s):
        return "—" if pd.isna(m) else f"{m:.3f} ± {s:.3f}"

    table = pd.DataFrame(
        {col: [fmt(m, s) for m, s in zip(mean[col], std[col])] for col in SUMMARY_METRICS},
        index=order,
    )
    table["Fit Seconds"] = runs.groupby("model", sort=False)["Fit Seconds"].mean().loc[order].map("{:.1f}".format)
    table.index.name = "Model"
    return table
