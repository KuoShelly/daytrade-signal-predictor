"""
資料來源介面層（Data Source Interface）
==========================================

背景
----
原始版本直接在 notebook 裡呼叫公司內部的行情資料平台（一個私有的 Python SDK，
內含連線 IP、帳號等內部資訊），這種寫法有兩個問題：

    1. 無法公開分享程式碼——內部 SDK 與連線資訊不應該出現在公開 repo。
    2. 資料存取邏輯跟後續的特徵工程/建模邏輯完全耦合在一起，
       換一個資料來源（例如改用其他供應商的 API，或是本機端的歷史 CSV）
       就要把整段程式碼重寫。

解法：定義一個抽象介面 `MarketDataSource`，把「拿到哪 5 張表」這件事
和「怎麼拿到這些表」拆開。正式環境可以實作一個接內部系統的
`InternalPlatformDataSource`（不放進這個公開 repo），
這裡提供 `MockMarketDataSource`，用亂數產生同樣欄位結構的假資料，
讓其他人可以直接 clone 下來跑通整條 pipeline，驗證程式邏輯本身是對的。

這是典型的 Adapter Pattern：pipeline.py 只依賴 MarketDataSource 這個抽象，
不在乎背後接的是內部系統還是假資料。
"""

from __future__ import annotations

import abc
from typing import List

import numpy as np
import pandas as pd

# 五張來源表的技術欄位（示意，實際特徵集依內部資料表而定）
_TECH_INDICATOR_COLS = ["K(9)", "D(9)", "RSI(5)", "RSI(10)", "DIF", "MACD", "DIF-MACD"]
_MA_COLS = ["MA5", "MA10", "MA20", "MA60"]
_MARGIN_COLS = ["融資餘額", "融券餘額", "資券比"]


class MarketDataSource(abc.ABC):
    """所有市場資料來源都要實作這 5 個方法，對應原始 5 張表。"""

    @abc.abstractmethod
    def get_technical_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        """日常用技術指標表：K、D、RSI、MACD 等。"""

    @abc.abstractmethod
    def get_daily_close_ranking(self, start_date: str, end_date: str) -> pd.DataFrame:
        """日收盤表排行：收盤價、成交量、振幅、成交量變動等。"""

    @abc.abstractmethod
    def get_moving_averages(self, start_date: str, end_date: str) -> pd.DataFrame:
        """日常用技術指標表（均線版）：MA5/10/20/60 等。"""

    @abc.abstractmethod
    def get_margin_trading(self, start_date: str, end_date: str) -> pd.DataFrame:
        """日融資券速選：融資融券相關欄位。"""

    @abc.abstractmethod
    def get_day_trade_volume(self, start_date: str, end_date: str) -> pd.DataFrame:
        """日當日沖銷交易：當沖成交量、非當沖成交量（label 計算會用到）。"""


class MockMarketDataSource(MarketDataSource):
    """
    用亂數產生資料，欄位結構跟正式環境一致，方便展示/測試 pipeline，
    不含任何真實市場資料或公司內部資訊。
    """

    def __init__(self, stock_ids: List[str] | None = None, seed: int = 42):
        self._stock_ids = stock_ids or [f"{1000 + i}" for i in range(50)]
        self._rng = np.random.default_rng(seed)

    def _date_range(self, start_date: str, end_date: str) -> List[str]:
        return [d.strftime("%Y%m%d") for d in pd.bdate_range(start_date, end_date)]

    def _base_frame(self, start_date: str, end_date: str) -> pd.DataFrame:
        dates = self._date_range(start_date, end_date)
        rows = [(d, s) for d in dates for s in self._stock_ids]
        return pd.DataFrame(rows, columns=["日期", "股票代號"])

    def get_technical_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._base_frame(start_date, end_date)
        for col in _TECH_INDICATOR_COLS:
            df[col] = self._rng.normal(50, 20, size=len(df))
        return df

    def get_daily_close_ranking(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._base_frame(start_date, end_date)
        df["收盤價"] = self._rng.uniform(10, 500, size=len(df))
        df["成交量"] = self._rng.integers(1_000, 200_000, size=len(df))
        df["振幅(%)"] = self._rng.uniform(0, 10, size=len(df))
        df["成交量變動(%)"] = self._rng.uniform(-30, 30, size=len(df))
        return df

    def get_moving_averages(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._base_frame(start_date, end_date)
        for col in _MA_COLS:
            df[col] = self._rng.uniform(10, 500, size=len(df))
        return df

    def get_margin_trading(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._base_frame(start_date, end_date)
        for col in _MARGIN_COLS:
            df[col] = self._rng.uniform(0, 1000, size=len(df))
        return df

    def get_day_trade_volume(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._base_frame(start_date, end_date)
        df["非當沖成交量"] = self._rng.integers(1_000, 150_000, size=len(df))
        df["當沖成交量"] = (df["非當沖成交量"] * self._rng.uniform(0, 0.6, size=len(df))).astype(int)
        return df


# 正式環境的做法（示意，不含實際內部連線資訊）：
#
# class InternalPlatformDataSource(MarketDataSource):
#     def __init__(self, host: str):
#         from internal_sdk import PlatformClient  # 內部套件，不對外公開
#         self._client = PlatformClient(host)
#
#     def get_technical_indicators(self, start_date, end_date):
#         return self._client.fetch("日常用技術指標表", start_date, end_date)
#     ...
