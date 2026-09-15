"""
Label 建構與資料合併
======================
把原本 notebook 裡混在一起的「抓資料 -> merge -> 算 label -> 加衍生特徵」
拆成三個獨立、可單獨測試的函式：

    merge_sources()          合併 5 張表成一張寬表
    build_day_trade_label()  定義「是否為當沖股」這個業務規則
    add_derived_flags()      加上高振幅/高成交量變動/高當沖比率等二元旗標

拆開的好處：這三個規則各自都是「業務邏輯」，之後如果分析師想調整
「當沖股」的定義（例如從 Top30 改成 Top50，或改用不同的成交量門檻），
只需要改 build_day_trade_label()，不會動到 merge 或特徵工程的程式碼，
也方便針對這條規則單獨寫單元測試（見 tests/test_labeling.py）。
"""

from __future__ import annotations

import pandas as pd

from .config import LABEL_CONFIG, FEATURE_CONFIG


def _coerce_numeric(df: pd.DataFrame, exclude_columns: list[str]) -> pd.DataFrame:
    """把非排除欄位以外的欄位轉成數值型態，對應原本的 trans_data()。"""
    numeric_columns = df.columns.difference(exclude_columns)
    df = df.copy()
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return df


def merge_sources(
    technical: pd.DataFrame,
    close_ranking: pd.DataFrame,
    moving_average: pd.DataFrame,
    margin_trading: pd.DataFrame,
    day_trade_volume: pd.DataFrame,
) -> pd.DataFrame:
    """依（日期, 股票代號）合併 5 張來源表，並去除合併鍵重複的資料列。"""
    merged = technical
    for other in (close_ranking, moving_average, margin_trading, day_trade_volume):
        merged = merged.merge(other, on=["日期", "股票代號"], how="inner")

    merged = _coerce_numeric(merged, exclude_columns=["股票代號", "日期"])
    merged = merged.drop_duplicates(subset=["日期", "股票代號"])
    return merged


def build_day_trade_label(
    day_trade_volume: pd.DataFrame,
    min_volume: int = LABEL_CONFIG.min_volume,
    top_n_per_day: int = LABEL_CONFIG.top_n_per_day,
) -> pd.DataFrame:
    """
    業務規則：每天成交量 > min_volume 的股票中，
    依「當沖成交量 / 總成交量」排序，取前 top_n_per_day 檔為正樣本（是否為當沖股=1）。

    回傳：['日期', '股票代號', '是否為當沖股', '當沖成交量']
    """
    df = _coerce_numeric(day_trade_volume, exclude_columns=["股票代號", "日期"])
    df["總成交量"] = df["當沖成交量"] + df["非當沖成交量"]
    df["當沖比例"] = df["當沖成交量"] / df["總成交量"]

    eligible = df.loc[df["總成交量"] > min_volume]
    top_n = (
        eligible.sort_values(["日期", "當沖比例"], ascending=[True, False])
        .groupby("日期")
        .head(top_n_per_day)
    )

    df["是否為當沖股"] = 0
    is_top_n = df.set_index(["日期", "股票代號"]).index.isin(
        top_n.set_index(["日期", "股票代號"]).index
    )
    df.loc[is_top_n, "是否為當沖股"] = 1

    return df[["日期", "股票代號", "是否為當沖股", "當沖成交量"]]


def add_derived_flags(
    df: pd.DataFrame,
    min_volume: int = LABEL_CONFIG.min_volume,
) -> pd.DataFrame:
    """加上高振幅/高成交量變動/高當沖比率三個二元旗標，並套用最低成交量門檻。"""
    df = df[df["成交量"] >= min_volume].copy()

    df["高振幅"] = (df["振幅(%)"] >= FEATURE_CONFIG.high_amplitude_pct).astype(int)
    df["成交量變動大"] = (
        df["成交量變動(%)"] >= FEATURE_CONFIG.high_volume_change_pct
    ).astype(int)
    df["當沖比例"] = df["當沖成交量"] / df["成交量"]
    df["高當沖比率"] = (
        df["當沖比例"] >= FEATURE_CONFIG.high_day_trade_ratio
    ).astype(int)

    return df
