"""
針對 build_day_trade_label() 這個核心業務規則寫的單元測試。

這是原本 notebook 完全沒有的部分——用互動式印出 DataFrame 人工目視檢查，
沒辦法在改動特徵工程或門檻值之後自動確認 label 邏輯還是對的。
這裡用小型、手算得出正確答案的假資料，驗證兩個規則:
    1. 成交量 <= min_volume 的股票，不論當沖比例多高都不能是正樣本
    2. 每天只留當沖比例最高的 top_n_per_day 檔
"""

import pandas as pd

from src.labeling import build_day_trade_label


def test_low_volume_stock_excluded_even_if_ratio_high():
    df = pd.DataFrame(
        {
            "日期": ["20240101", "20240101"],
            "股票代號": ["1101", "1102"],
            "當沖成交量": [500, 5000],
            "非當沖成交量": [500, 5000],  # 兩者當沖比例都是 0.5，但總量都 <= 10000
        }
    )
    result = build_day_trade_label(df, min_volume=10_000, top_n_per_day=1)
    assert result["是否為當沖股"].sum() == 0


def test_top_n_per_day_respected():
    df = pd.DataFrame(
        {
            "日期": ["20240101"] * 3,
            "股票代號": ["1101", "1102", "1103"],
            "當沖成交量": [9000, 6000, 3000],
            "非當沖成交量": [1000, 4000, 7000],  # 當沖比例分別為 0.9, 0.6, 0.3
        }
    )
    result = build_day_trade_label(df, min_volume=5_000, top_n_per_day=2)
    positive = result.loc[result["是否為當沖股"] == 1, "股票代號"].tolist()
    assert sorted(positive) == ["1101", "1102"]  # 當沖比例最高的兩檔
