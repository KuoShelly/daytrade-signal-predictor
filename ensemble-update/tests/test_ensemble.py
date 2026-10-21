"""
模型融合的單元測試（重構後新增）。

重點不是分數，而是兩個正確性保證：
    1. Stacking 的 OOF 切分不會用到未來資料（每一折的訓練日期都早於預測日期）。
    2. Average 真的是成員機率的平均；Stacking 能跑完整流程並輸出合法機率。
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ensemble import AverageEnsemble, StackingEnsemble, time_ordered_folds
from src.model_comparison import LightGBMCandidate, SklearnCandidate, prepare_comparison_data
from src.pipeline import load_and_prepare
from src.synthetic_data import SyntheticSignalDataSource


@pytest.fixture(scope="module")
def data():
    source = SyntheticSignalDataSource(stock_ids=[f"{3000 + i}" for i in range(60)], seed=11)
    return prepare_comparison_data(load_and_prepare(source, "2024-01-01", "2024-05-31"))


@pytest.fixture(scope="module")
def members():
    return {
        "lr": lambda: SklearnCandidate(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        "rf": lambda: SklearnCandidate(RandomForestClassifier(n_estimators=20, random_state=0)),
        "lgb": lambda: LightGBMCandidate(),
    }


def test_time_ordered_folds_never_train_on_future():
    dates = np.repeat(np.array([f"2024{m:02d}01" for m in range(1, 11)], dtype=object), 3)  # 10 天、每天 3 筆，與 pipeline 一樣是 object 字串
    folds = time_ordered_folds(dates, n_folds=4)
    assert len(folds) == 4
    for fit_mask, predict_mask in folds:
        assert fit_mask.any() and predict_mask.any()
        assert dates[fit_mask].max() < dates[predict_mask].min()
        assert not (fit_mask & predict_mask).any()
    # 第一段永遠只當訓練資料，其餘每一天剛好被預測一次
    predicted = np.sum([p for _, p in folds], axis=0)
    assert predicted[:6].sum() == 0 and (predicted[6:] == 1).all()


def test_average_is_mean_of_members(data, members):
    ens = AverageEnsemble(members).fit(data.X_train, data.y_train, data.train_dates, data.feature_names)
    member_scores = ens.member_predictions(data.X_val)
    np.testing.assert_allclose(ens.predict_proba(data.X_val), np.mean(list(member_scores.values()), axis=0))


def test_stacking_outputs_valid_probabilities(data, members):
    ens = StackingEnsemble(members, n_folds=3).fit(
        data.X_train, data.y_train, data.train_dates, data.feature_names
    )
    proba = ens.predict_proba(data.X_val)
    assert proba.shape == data.y_val.shape
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(ens.meta_weights) == {"lr", "rf", "lgb"}


def test_ensemble_model_runs_through_main_pipeline():
    # 融合模型要能直接替換 DayTradeModel，走完 train -> evaluate -> predict
    from src.ensemble_model import EnsembleDayTradeModel
    from src.pipeline import run_training_pipeline

    source = SyntheticSignalDataSource(stock_ids=[f"{4000 + i}" for i in range(40)], seed=5)
    model, metrics, confusion, daily = run_training_pipeline(
        source, "2024-01-01", "2024-04-30", model=EnsembleDayTradeModel(method="average")
    )
    assert 0.5 < metrics["AUC-ROC"] <= 1
    assert sum(confusion.values()) == daily["總股票數"].sum()

    raw = load_and_prepare(source, "2024-01-01", "2024-04-30")
    pred = model.predict(raw)
    assert pred["預測分數"].between(0, 1).all()
