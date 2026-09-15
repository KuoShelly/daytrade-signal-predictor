"""
CLI 進入點
===========
原本 notebook 是直接在 cell 裡寫死本機路徑（C:\\Users\\user1\\...）去讀 CSV，
只能在原作者的電腦上跑。這裡改成標準的命令列介面，任何人 clone 下來後：

    pip install -r requirements.txt
    python train.py --start-date 2023-12-20 --end-date 2024-12-31

就能跑通整條 pipeline（預設用 MockMarketDataSource 產生的示範資料；
接上真實資料只需要另外實作一個 MarketDataSource 並在這裡替換掉即可）。
"""

import argparse

from src.data_interface import MockMarketDataSource
from src.pipeline import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="當沖熱門股預測 - 訓練與驗證")
    parser.add_argument("--start-date", default="2023-12-20")
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()

    source = MockMarketDataSource()
    model, metrics, confusion, daily = run_training_pipeline(source, args.start_date, args.end_date)

    print("\n=== 整體績效指標（示範資料，僅供驗證流程用） ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== 混淆矩陣 ===")
    for k, v in confusion.items():
        print(f"{k}: {v}")

    print("\n=== 每日預測摘要（前 5 天） ===")
    print(daily.head())


if __name__ == "__main__":
    main()
