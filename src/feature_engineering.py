"""
滯後特徵工程（Lagged Feature Engineering）
=============================================
從原本 ImprovedDayTradePredictor 裡拆出來，職責單一化：
「給一張已經清理過的寬表，回傳加了 n_lags 天滯後特徵的版本」。

保留原本的關鍵設計：
    1. 只保留有 > n_lags 天歷史資料的股票，避免新上市/資料不足的股票
       在 shift() 之後全是 NaN。
    2. 缺失值只用「前向填充」，不用整體平均數/中位數去補，
       避免用到未來資訊（look-ahead bias）。
    3. 訓練時記住 feature_cols 的順序，預測時強制使用同一份順序，
       這是這支程式最重要的正確性保證——訓練/推論的特徵欄位如果對不齊，
       模型會用錯的意義去解讀每一欄數字，而且不會報錯，是最隱蔽的 bug 來源。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


class LaggedFeatureBuilder:
    def __init__(self, n_lags: int = 3):
        self.n_lags = n_lags
        self.feature_cols: Optional[List[str]] = None
        self.lagged_feature_cols: Optional[List[str]] = None

    def fit_transform(self, df: pd.DataFrame, label_col: str = "是否為當沖股") -> pd.DataFrame:
        """訓練階段呼叫：從資料自動決定要用哪些數值型欄位當特徵。"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.feature_cols = sorted(
            col for col in numeric_columns if col not in (label_col, "日期")
        )
        self.lagged_feature_cols = [
            f"{col}_lag{lag}" for col in self.feature_cols for lag in range(1, self.n_lags + 1)
        ]
        return self._build(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """推論階段呼叫：沿用 fit_transform 決定好的特徵欄位，不重新推導。"""
        if self.feature_cols is None:
            raise RuntimeError("尚未呼叫過 fit_transform()，無法決定特徵欄位")
        return self._build(df)

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["日期", "股票代號"]).reset_index(drop=True)

        stock_date_counts = df.groupby("股票代號")["日期"].nunique()
        valid_stocks = stock_date_counts[stock_date_counts > self.n_lags].index
        if len(valid_stocks) == 0:
            raise ValueError(f"沒有股票有超過 {self.n_lags} 天的數據")

        df_filtered = df[df["股票代號"].isin(valid_stocks)].copy()

        grouped = df_filtered.groupby("股票代號")
        lagged = {
            f"{col}_lag{lag}": grouped[col].shift(lag)
            for col in self.feature_cols
            for lag in range(1, self.n_lags + 1)
        }
        df_with_lags = pd.concat([df_filtered, pd.DataFrame(lagged, index=df_filtered.index)], axis=1)

        # 只做前向填充，剩餘的（一支股票最早期的資料）補 0，避免引入未來資訊
        df_with_lags = df_with_lags.ffill().fillna(0)

        # 每支股票移除前 n_lags 天（滯後特徵還沒填滿的資料）
        valid_rows = []
        for stock in valid_stocks:
            stock_data = df_with_lags[df_with_lags["股票代號"] == stock]
            stock_dates = sorted(stock_data["日期"].unique())
            valid_start_date = stock_dates[self.n_lags]
            valid_rows.append(stock_data[stock_data["日期"] >= valid_start_date])

        return pd.concat(valid_rows, ignore_index=True)
