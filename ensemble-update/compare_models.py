"""
模型比較實驗 CLI（重構後新增，使用 Mock / 合成資料）
=======================================================

    python compare_models.py                      # 兩種情境都跑，5 個 seed
    python compare_models.py --scenario signal --seeds 0 1 2 3 4

兩種資料情境：
    noise   MockMarketDataSource：所有欄位都是獨立亂數，沒有任何可預測的訊號。
            用途是「洩漏檢查」——所有模型的 AUC 都應該在 0.5 附近，
            如果某個模型明顯高於 0.5，代表特徵工程或切分有 look-ahead bias。
    signal  SyntheticSignalDataSource：人為埋入已知訊號（見 src/synthetic_data.py），
            用來檢查模型能不能抓到「確定存在」的訊號，以及不同模型抓訊號的能力差異。

注意：兩種情境都不是真實市場資料，跑出來的數字不代表模型在真實台股上的表現。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data_interface import MockMarketDataSource
from src.model_comparison import run_model_comparison, summarize
from src.synthetic_data import SyntheticSignalDataSource

SCENARIOS = {
    "noise": ("Pure-noise Mock data (leakage check)", lambda seed: MockMarketDataSource(seed=seed)),
    "signal": ("Synthetic data with a planted signal", lambda seed: SyntheticSignalDataSource(seed=seed)),
}

DISCLAIMER = (
    "> **Mock / synthetic data only.** These numbers come from randomly generated data and are\n"
    "> NOT real market results. They only verify that the comparison pipeline works.\n"
)


def main():
    parser = argparse.ArgumentParser(description="LightGBM vs Logistic Regression vs Random Forest（Mock 資料）")
    parser.add_argument("--start-date", default="2023-12-20")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--scenario", choices=["noise", "signal", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["noise", "signal"] if args.scenario == "both" else [args.scenario]
    md_parts = ["# Model Comparison Results (Mock data)\n", DISCLAIMER]

    for key in scenarios:
        title, factory = SCENARIOS[key]
        print(f"\n=== {title} | seeds={args.seeds} ===")
        runs = run_model_comparison(factory, args.start_date, args.end_date, args.seeds)
        runs.to_csv(out_dir / f"model_comparison_{key}_runs.csv", index=False, encoding="utf-8-sig")

        table = summarize(runs)
        print(table.to_string())

        first = runs.iloc[0]
        md_parts += [
            f"\n## {title}\n",
            f"- Train period: {first['train_period']}, validation period: {first['val_period']}",
            f"- Train rows: {first['n_train']:,}, validation rows: {first['n_val']:,}, "
            f"validation positive ratio: {runs['val_pos_ratio'].mean():.1%}",
            f"- Seeds: {args.seeds} (mean ± std across seeds)\n",
            table.to_markdown(),
            "",
        ]

    (out_dir / "model_comparison.md").write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\n結果已寫入 {out_dir}/（Mock 資料，非真實市場結果）")


if __name__ == "__main__":
    main()
