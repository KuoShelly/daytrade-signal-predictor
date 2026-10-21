"""
模型比較實驗的單元測試（重構後新增）。

驗證的是「比較流程本身是否公平、正確」，不是模型分數高低：
    1. 時間切分：驗證集的每一天都在訓練集之後。
    2. 所有模型在同一批驗證資料上評估，且都產出合法的指標。
    3. 評估函式的每日 Top-N 命中率用手算得出答案的小資料驗證。
    4. 合成資料來源：同 seed 可重現，且埋入的訊號確實存在（sanity check）。
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.model_comparison import (
    LaggedRatioBaseline,
    LightGBMCandidate,
    SklearnCandidate,
    compare_models,
    evaluate_scores,
    prepare_comparison_data,
)
from src.pipeline import load_and_prepare
from src.synthetic_data import SyntheticSignalDataSource

START, END = "2024-01-01", "2024-04-30"


@pytest.fixture(scope="module")
def comparison_data():
    source = SyntheticSignalDataSource(stock_ids=[f"{2000 + i}" for i in range(60)], seed=7)
    return prepare_comparison_data(load_and_prepare(source, START, END))


def test_validation_dates_strictly_after_training_dates(comparison_data):
    assert comparison_data.train_dates.max() < comparison_data.val_dates.min()


def test_all_candidates_evaluated_on_same_validation_set(comparison_data):
    candidates = {
        "baseline": lambda: LaggedRatioBaseline(),
        "rf": lambda: SklearnCandidate(RandomForestClassifier(n_estimators=20, random_state=0)),
        "lgb": lambda: LightGBMCandidate(),
    }
    result = compare_models(comparison_data, candidates)

    assert list(result["model"]) == ["baseline", "rf", "lgb"]
    assert result["AUC-ROC"].between(0, 1).all()
    assert result["Top-30 Hit Rate"].between(0, 1).all()
    # baseline 沒有機率輸出，門檻型指標應該是 NaN；其他模型的混淆矩陣總和 = 驗證集筆數
    assert np.isnan(result.loc[0, "Accuracy"])
    n_val = len(comparison_data.y_val)
    for _, row in result.iloc[1:].iterrows():
        assert row["TP"] + row["FP"] + row["TN"] + row["FN"] == n_val


def test_top_n_hit_rate_hand_computed():
    # 兩天、每天取 Top-2：第一天前兩名都對（1.0），第二天前兩名對一個（0.5）→ 平均 0.75
    dates = np.array(["d1"] * 3 + ["d2"] * 3)
    y_true = np.array([1, 1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.1, 0.9, 0.8, 0.1])
    metrics = evaluate_scores(y_true, scores, dates, threshold=0.5, top_n=2)
    assert metrics["Top-2 Hit Rate"] == pytest.approx(0.75)
    assert metrics["Daily Hit Rate"] == pytest.approx((2 / 2 + 1 / 2) / 2)


def test_synthetic_source_is_reproducible():
    a = SyntheticSignalDataSource(seed=3).get_day_trade_volume(START, END)
    b = SyntheticSignalDataSource(seed=3).get_day_trade_volume(START, END)
    pd.testing.assert_frame_equal(a, b)


def test_planted_signal_is_detectable(comparison_data):
    # 合成資料埋了「當沖傾向的時間持續性」，昨天的當沖比例應該明顯優於亂猜
    baseline = LaggedRatioBaseline().fit(None, None, None, comparison_data.feature_names)
    scores = baseline.predict_proba(comparison_data.X_val)
    metrics = evaluate_scores(comparison_data.y_val, scores, comparison_data.val_dates, threshold=None)
    assert metrics["AUC-ROC"] > 0.7
