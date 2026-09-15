"""
模型層：LightGBM 訓練 / 預測 / 評估
======================================

這裡對應原本的 ImprovedDayTradePredictor（重構前 notebook 裡還留著一個
沒有刪掉的舊版 DayTradePredictor，功能重複但邏輯較舊，重構時直接移除，
只保留這個「已驗證效果較好」的版本）。

改動重點：
    - 特徵工程（LaggedFeatureBuilder）與資料切分（time_based_split）抽成獨立模組，
      這支檔案只負責「拿到特徵矩陣之後」的事：標準化、訓練、預測、評估。
    - class_imbalance 的處理維持原本的 scale_pos_weight 作法
      （驗證集正樣本比例約 25-32%，中度不平衡，用 scale_pos_weight 比直接上
      SMOTE 更適合這種「比例本身有business意義、不希望合成樣本汙染分布」的情境）。
"""

from __future__ import annotations

from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .config import LGB_CONFIG, MODEL_CONFIG
from .feature_engineering import LaggedFeatureBuilder


def time_based_split(df: pd.DataFrame, validation_ratio: float = MODEL_CONFIG.validation_ratio):
    """依日期切分，避免用未來的資料驗證過去的模型（look-ahead bias）。"""
    dates = sorted(df["日期"].unique())
    split_idx = int(len(dates) * (1 - validation_ratio))
    train_dates, val_dates = dates[:split_idx], dates[split_idx:]
    return df[df["日期"].isin(train_dates)], df[df["日期"].isin(val_dates)]


class DayTradeModel:
    def __init__(self, n_lags: int = MODEL_CONFIG.n_lags):
        self.feature_builder = LaggedFeatureBuilder(n_lags=n_lags)
        self.scaler = StandardScaler()
        self.model: Optional[lgb.Booster] = None

    def _prepare_features(self, df: pd.DataFrame, is_training: bool):
        X = df[self.feature_builder.lagged_feature_cols]
        X_scaled = self.scaler.fit_transform(X) if is_training else self.scaler.transform(X)
        y = df["是否為當沖股"].values if "是否為當沖股" in df.columns else None
        return X_scaled, y

    def train(self, df: pd.DataFrame, validation_ratio: float = MODEL_CONFIG.validation_ratio):
        df_processed = self.feature_builder.fit_transform(df)
        train_df, val_df = time_based_split(df_processed, validation_ratio)

        X_train, y_train = self._prepare_features(train_df, is_training=True)
        X_val, y_val = self._prepare_features(val_df, is_training=False)

        train_pos_ratio = y_train.mean()
        scale_pos_weight = 1.0 / train_pos_ratio if train_pos_ratio > 0 else 1.0

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.model = lgb.train(
            LGB_CONFIG.to_lgb_params(scale_pos_weight),
            train_data,
            num_boost_round=LGB_CONFIG.num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                early_stopping(stopping_rounds=LGB_CONFIG.early_stopping_rounds),
                log_evaluation(period=100),
            ],
        )
        return train_df, val_df

    def predict(self, df: pd.DataFrame, threshold: float = MODEL_CONFIG.predict_threshold) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("模型尚未訓練")

        df_processed = self.feature_builder.transform(df)
        X, _ = self._prepare_features(df_processed, is_training=False)
        proba = self.model.predict(X)

        result = df_processed[["日期", "股票代號"]].copy()
        result["預測分數"] = proba
        result["預測是否為當沖股"] = (proba >= threshold).astype(int)
        return result

    def evaluate(self, df_processed: pd.DataFrame, threshold: float = MODEL_CONFIG.predict_threshold):
        """
        評估已經含有滯後特徵的資料（例如 train() 回傳的 val_df）。

        注意：這裡刻意「不」重新呼叫 feature_builder.transform()。
        原始 notebook 版本在這裡對已經 lag 過的 val_df 又跑了一次
        create_lagged_features，等於在已經位移過的資料上再位移一次，
        這裡修正這個問題：驗證資料在 train() 階段已經處理過，直接用即可；
        真正要對「全新、尚未處理」的原始資料做特徵工程，請用 predict()。
        """
        if self.model is None:
            raise RuntimeError("模型尚未訓練")

        X, y_true = self._prepare_features(df_processed, is_training=False)
        y_proba = self.model.predict(X)
        y_pred = (y_proba >= threshold).astype(int)

        overall_metrics = {
            "整體準確率": accuracy_score(y_true, y_pred),
            "精確率": precision_score(y_true, y_pred, zero_division=0),
            "召回率": recall_score(y_true, y_pred, zero_division=0),
            "F1分數": f1_score(y_true, y_pred, zero_division=0),
            "AUC-ROC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        }

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        confusion = {"真陽性(TP)": int(tp), "假陽性(FP)": int(fp), "真陰性(TN)": int(tn), "假陰性(FN)": int(fn)}

        df_processed = df_processed.assign(預測分數=y_proba, 預測結果=y_pred)
        daily = df_processed.groupby("日期").agg(
            實際當沖股數=("是否為當沖股", "sum"),
            總股票數=("是否為當沖股", "count"),
            預測當沖股數=("預測結果", "sum"),
        )
        daily["正確預測數"] = df_processed.groupby("日期").apply(
            lambda x: ((x["預測結果"] == 1) & (x["是否為當沖股"] == 1)).sum(),
            include_groups=False,
        )
        daily["每日命中率"] = daily["正確預測數"] / daily["實際當沖股數"].replace(0, np.nan)

        return overall_metrics, confusion, daily

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("模型尚未訓練")
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_builder.lagged_feature_cols,
                    "importance": self.model.feature_importance(),
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
