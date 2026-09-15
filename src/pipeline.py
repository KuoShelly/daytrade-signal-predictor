"""
Pipeline：串起 資料來源 -> 合併/標籤 -> 特徵工程 -> 訓練 -> 評估
==================================================================
這支檔案是唯一知道「完整流程長什麼樣子」的地方，其他模組互相都不依賴，
只依賴 config 和彼此的輸入輸出介面。
"""

from __future__ import annotations

from .data_interface import MarketDataSource
from .labeling import add_derived_flags, build_day_trade_label, merge_sources
from .model import DayTradeModel


def load_and_prepare(source: MarketDataSource, start_date: str, end_date: str):
    """呼叫資料來源、合併、建 label、加衍生特徵，回傳可直接餵給模型的寬表。"""
    technical = source.get_technical_indicators(start_date, end_date)
    close_ranking = source.get_daily_close_ranking(start_date, end_date)
    moving_average = source.get_moving_averages(start_date, end_date)
    margin_trading = source.get_margin_trading(start_date, end_date)
    day_trade_volume = source.get_day_trade_volume(start_date, end_date)

    label_df = build_day_trade_label(day_trade_volume)

    merged = merge_sources(technical, close_ranking, moving_average, margin_trading, day_trade_volume)
    # merged 已經從 day_trade_volume 帶有「當沖成交量」欄位，這裡只需要併入 label 本身，
    # 避免同一欄位重複合併時被 pandas 自動加上 _x/_y 後綴
    merged = merged.merge(label_df[["日期", "股票代號", "是否為當沖股"]], on=["日期", "股票代號"], how="inner")
    merged = add_derived_flags(merged)

    return merged


def run_training_pipeline(source: MarketDataSource, start_date: str, end_date: str):
    """完整跑一次：載入資料 -> 訓練 -> 在驗證集上評估，回傳 model 與評估結果。"""
    df = load_and_prepare(source, start_date, end_date)

    model = DayTradeModel()
    _, val_df = model.train(df)
    metrics, confusion, daily = model.evaluate(val_df)

    return model, metrics, confusion, daily
