"""
帶有「已知訊號」的合成資料來源（僅供模型比較實驗使用）
==========================================================

為什麼需要這支檔案
------------------
`MockMarketDataSource` 的每一個欄位都是獨立亂數，今天的特徵跟明天的 label
之間沒有任何關係。用它來比較模型，所有模型的 AUC 都會落在 0.5 附近，
比不出任何東西——這本身是有用的「洩漏檢查」（見 README），但無法回答
「不同模型抓訊號的能力是否不同」。

這裡的 `SyntheticSignalDataSource` 刻意「埋」進一個已知、可控的訊號，
讓模型比較有東西可以比。訊號設計如下（全部是人為設定，不代表真實市場）：

    1. 每檔股票有一個固定的「當沖傾向」mu_s（股票層級的持續性）。
    2. 當沖傾向的潛在變數 z 是 AR(1)：今天的 z 跟昨天的 z 相關（時間持續性）。
    3. 昨天 RSI(5) 落在極端區（> 75 或 < 25）時，今天的 z 會被往上推
       ——這是一個「非線性」的門檻效果，線性模型不容易直接表達。
    4. 當天振幅與 z 正相關，所以振幅的滯後值也帶有部分資訊。

當沖比例 = 0.6 * sigmoid(z)，再反推出當沖成交量；成交量 = 當沖 + 非當沖，
讓 label 規則（總成交量門檻）與衍生特徵用到的成交量欄位一致。

重要提醒
--------
因為訊號是我們自己設計的，「哪個模型比較好」在很大程度上是由這個生成機制
決定的（例如第 3 點天生對樹模型有利）。這支資料來源只能用來驗證
比較實驗的流程是否正確、模型是否抓得到「已知存在」的訊號，
**不能**用來推論哪個模型在真實台股資料上表現較好。
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd

from .data_interface import MockMarketDataSource, _TECH_INDICATOR_COLS


class SyntheticSignalDataSource(MockMarketDataSource):
    """在 Mock 資料上埋入已知的時間持續性 + 非線性訊號。"""

    def __init__(
        self,
        stock_ids: List[str] | None = None,
        seed: int = 42,
        persistence: float = 0.6,
        stock_effect_std: float = 0.8,
        rsi_extreme_boost: float = 0.6,
        noise_std: float = 0.5,
    ):
        # 預設 100 檔：每天取 Top30，正樣本比例約 30%，
        # 接近原始版本驗證集 25-32% 的不平衡程度（50 檔時會變成 60% 正樣本）
        super().__init__(stock_ids=stock_ids or [f"{1000 + i}" for i in range(100)], seed=seed)
        self._seed = seed
        self._persistence = persistence
        self._stock_effect_std = stock_effect_std
        self._rsi_extreme_boost = rsi_extreme_boost
        self._noise_std = noise_std

    @lru_cache(maxsize=4)
    def _panel(self, start_date: str, end_date: str) -> pd.DataFrame:
        """一次產生整個 (日期 x 股票) 面板，確保 5 張表彼此一致。"""
        rng = np.random.default_rng(self._seed)
        dates = self._date_range(start_date, end_date)
        n_days, n_stocks = len(dates), len(self._stock_ids)

        mu = rng.normal(0.0, self._stock_effect_std, size=n_stocks)
        rsi5 = np.clip(rng.normal(50, 20, size=(n_days, n_stocks)), 0, 100)
        extreme = ((rsi5 > 75) | (rsi5 < 25)).astype(float)

        z = np.empty((n_days, n_stocks))
        z[0] = mu + rng.normal(0, self._noise_std, size=n_stocks)
        for t in range(1, n_days):
            z[t] = (
                mu
                + self._persistence * (z[t - 1] - mu)
                + self._rsi_extreme_boost * extreme[t - 1]
                + rng.normal(0, self._noise_std, size=n_stocks)
            )

        ratio = 0.6 / (1.0 + np.exp(-z))  # 當沖比例落在 (0, 0.6)
        non_day_trade = rng.integers(1_000, 150_000, size=(n_days, n_stocks))
        day_trade = (non_day_trade * ratio / (1.0 - ratio)).astype(int)
        amplitude = np.clip(4.0 + 1.5 * z + rng.normal(0, 1.5, size=z.shape), 0, 10)

        panel = pd.DataFrame(
            {
                "日期": np.repeat(dates, n_stocks),
                "股票代號": np.tile(self._stock_ids, n_days),
                "RSI(5)": rsi5.ravel(),
                "振幅(%)": amplitude.ravel(),
                "非當沖成交量": non_day_trade.ravel(),
                "當沖成交量": day_trade.ravel(),
            }
        )
        panel["成交量"] = panel["非當沖成交量"] + panel["當沖成交量"]
        return panel

    def get_technical_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = super().get_technical_indicators(start_date, end_date)  # 其餘指標維持純雜訊
        df["RSI(5)"] = self._panel(start_date, end_date)["RSI(5)"].to_numpy()
        return df[["日期", "股票代號", *_TECH_INDICATOR_COLS]]

    def get_daily_close_ranking(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = super().get_daily_close_ranking(start_date, end_date)
        panel = self._panel(start_date, end_date)
        df["成交量"] = panel["成交量"].to_numpy()
        df["振幅(%)"] = panel["振幅(%)"].to_numpy()
        return df

    def get_day_trade_volume(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self._panel(start_date, end_date)[["日期", "股票代號", "非當沖成交量", "當沖成交量"]].copy()
