"""
把融合模型接進主 pipeline（重構後新增）
==========================================

EnsembleDayTradeModel 繼承 DayTradeModel，沿用同一套
特徵工程（LaggedFeatureBuilder）、時間切分（time_based_split）、
predict() / evaluate() 介面，只把「LightGBM booster」換成融合模型。
因此 train.py 只要換一個類別，就能用融合模型跑完整流程。

與 DayTradeModel 的差異：
    - 驗證集不參與任何訓練決策（成員 LightGBM 的 early stopping 只用訓練期間內部切出的資料）。
    - 沒有 feature_importance()；Stacking 改提供 meta_weights 看各成員的權重。
"""

from __future__ import annotations

import pandas as pd

from .config import MODEL_CONFIG
from .ensemble import AverageEnsemble, StackingEnsemble
from .model import DayTradeModel, time_based_split
from .model_comparison import base_model_factories


class _ProbaAdapter:
    """讓融合模型符合 DayTradeModel 預期的 self.model.predict(X) -> 機率 介面。"""

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def predict(self, X):
        return self.ensemble.predict_proba(X)


class EnsembleDayTradeModel(DayTradeModel):
    METHODS = {"stacking": StackingEnsemble, "average": AverageEnsemble}

    def __init__(self, method: str = "stacking", n_lags: int = MODEL_CONFIG.n_lags, random_state: int = 42):
        if method not in self.METHODS:
            raise ValueError(f"method 必須是 {list(self.METHODS)} 之一")
        super().__init__(n_lags=n_lags)
        self.method = method
        self.random_state = random_state
        self.ensemble = None

    def train(self, df: pd.DataFrame, validation_ratio: float = MODEL_CONFIG.validation_ratio):
        df_processed = self.feature_builder.fit_transform(df)
        train_df, val_df = time_based_split(df_processed, validation_ratio)

        X_train, y_train = self._prepare_features(train_df, is_training=True)
        members = base_model_factories(self.random_state)
        self.ensemble = self.METHODS[self.method](members).fit(
            X_train,
            y_train,
            train_df["日期"].to_numpy(),
            list(self.feature_builder.lagged_feature_cols),
        )
        self.model = _ProbaAdapter(self.ensemble)
        return train_df, val_df

    @property
    def meta_weights(self):
        return getattr(self.ensemble, "meta_weights", None)

    def feature_importance(self, top_n: int = 20):
        raise NotImplementedError("融合模型沒有單一的 feature importance；Stacking 請看 meta_weights")
