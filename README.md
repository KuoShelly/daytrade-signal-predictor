# Day-Trade Signal Predictor

Uses multi-source Taiwan stock data — intraday technical indicators, closing rankings, moving averages, margin trading and short selling, and day-trading volume — to predict "stocks whose day-trading share of volume will be high tomorrow," helping traders narrow down what they need to watch each day.

> **About the data source**: The original version connected to a private SDK for an internal market-data platform (including connection details).
> This public version abstracts data access into a `MarketDataSource` interface and provides
> `MockMarketDataSource`, which generates demo data with the same column structure, so anyone can
> `clone` the repo, run the entire pipeline, and verify that the program logic itself is correct.
> Read "3. Engineering Design Decisions," item 1, to understand why it was split this way.

---

## 1. Business Problem

Day-trading activity in Taiwan stocks is concentrated in a small number of stocks, but there are over a thousand securities every day. Manually screening for stocks whose "day-trading activity will be high today"
is very time-consuming, and it is easy to miss stocks whose intraday signals have not yet clearly developed but whose technicals are already showing signs.

**Goal**: Use technical indicators and chip (positioning) data from the previous trading day (and earlier) to predict the list of stocks whose "day-trading ratio will rank in
the top 30 of the entire market today," so traders can prioritize this list instead of scanning the whole market one stock at a time.

**Label definition** (`src/labeling.py`):
- Daily volume > 10,000 shares (to avoid interference from stocks with poor liquidity)
- Among qualifying stocks, rank by "day-trading volume / total volume" and take the top 30 each day as positive samples

## 2. Pipeline Architecture

```
MarketDataSource (5 source tables)
        |
        v
merge_sources()          Merge into a wide table by (date, stock ticker), deduplicate, convert to numeric types
        |
        v
build_day_trade_label()  Compute the "is day-trading stock" label
add_derived_flags()      Add high-amplitude / high-volume-change / high-day-trading-ratio flags
        |
        v
LaggedFeatureBuilder     Group by stock and build n=3 day lagged features
                         (keep only stocks with enough history, to avoid all-NaN rows for new listings)
        |
        v
time_based_split()       Split train / validation by date (never random split,
                         to avoid validating a past model with future data)
        |
        v
DayTradeModel (LightGBM) Handle class imbalance with scale_pos_weight -> train -> evaluate
```

Mapping to source files:

| File | Responsibility |
|---|---|
| `src/data_interface.py` | Abstract data-source interface + Mock implementation |
| `src/labeling.py` | Label computation, multi-table merging, business-rule flags |
| `src/feature_engineering.py` | Lagged feature construction (fit/transform separated) |
| `src/model.py` | LightGBM training / prediction / evaluation |
| `src/pipeline.py` | Connects all of the above steps |
| `train.py` | CLI entry point |
| `tests/test_labeling.py` | Unit tests for label business rules |

## 3. Engineering Design Decisions (Trade-offs Made During Refactoring)

### 1. Why extract a `MarketDataSource` interface?
The original notebook put "calling the internal system API" and "feature engineering/modeling" in the same cell,
directly coupling the company's private internal SDK and connection details. It could not be shared publicly, nor could it be tested
for correct program logic in an environment without internal network access. After extracting the abstract interface, `pipeline.py`
has no idea whether it is backed by the internal system or fake data. This is the standard Adapter Pattern,
solving three problems at once: swapping data sources, writing tests, and sharing the code publicly.

### 2. Why must train/validation be split by time?
If financial time-series data is split randomly with `train_test_split(random_state=...)`,
the validation set gets mixed with data from the training set's "future." The model effectively peeks at the future before validation,
and performance numbers become severely inflated. `time_based_split()` guarantees the validation set is always dated after the training set.

### 3. Why use forward fill for lagged features instead of the mean?
After building lagged features, `LaggedFeatureBuilder` uses only `ffill()`, not the overall mean/median.
Filling with the mean leaks "statistics that can only be computed in the future" into current features (data leakage);
forward fill uses only information that "has already happened" for that stock, matching the real trading constraint of "only using currently known data."

### 4. Fixed an easy-to-miss bug from the original version
The original notebook's `evaluate()` called `create_lagged_features()` again on a validation set that "had already been processed with lagged features during `train()`,"
effectively shifting already-shifted data a second time. This refactored version separates the two scenarios of "the validation set prepared during training"
and "inference on brand-new raw data" (`evaluate()` takes processed data; only `predict()` reruns feature engineering),
avoiding this hidden double-shift problem.

### 5. Class imbalance: `scale_pos_weight` instead of SMOTE
The positive-sample ratio is about 25-32% (depending on the date), a moderate imbalance. LightGBM's built-in
`scale_pos_weight` was chosen over SMOTE-style oversampling because the positive ratio itself has a clear business meaning
(the definition of "top 30 by day-trading ratio"). Synthetic oversampled samples could distort this distribution;
weight adjustment fits this situation better.

## 4. Test Results from the Original Version (For Reference Only)

The numbers below come from the original notebook run on real internal data (2023/12/20 - 2024/12/31,
80/20 train/validation split by date). **Because the data source in this public refactored version has been replaced
with Mock data, these numbers cannot be reproduced** — they are listed here to show that the model
is genuinely effective on real data, and the refactoring itself did not change the model logic:

| Metric | Value |
|---|---|
| Accuracy | 0.9034 |
| Precision | 0.8107 |
| Recall | 0.8835 |
| F1 Score | 0.8455 |
| AUC-ROC | 0.9622 |
| Average Daily Hit Rate | 0.8822 |

Confusion matrix (validation set, 3,385 rows total): TP 895 / FP 209 / TN 2,163 / FN 118

Top feature importances (from the original experiment): `當沖比例_lag1` (day-trading ratio), `週轉率(%)_lag1` (turnover rate),
`相對強弱比(日)_lag1` (daily relative strength) — the most recent day's day-trading ratio and turnover rate are the most important predictive signals,
consistent with the market intuition that "day-trading heat has short-term persistence."

## 5. How to Run

```bash
pip install -r requirements.txt
python train.py --start-date 2024-01-01 --end-date 2024-06-30
pytest tests/
```

`MockMarketDataSource` is used by default to generate demo data. To connect a real data source,
simply implement a new `MarketDataSource` subclass (see the example code at the bottom of `data_interface.py`)
and replace `MockMarketDataSource()` in `train.py`.
The rest of the pipeline requires no changes at all.

## 6. Known Limitations and Future Directions

- Currently uses a single time-window train/validation split; walk-forward
  cross-validation has not been done yet, so model stability has only been verified on one validation window.
- Feature importance shows the model relies heavily on `_lag1` features, meaning that when market conditions change quickly,
  the model may need more frequent retraining; "how often to retrain" has not yet been quantified.
- There is currently no simulation of transaction costs, slippage, or practical executability (liquidity impact).
  High prediction accuracy does not directly mean a profitable strategy. This is a common gap between model evaluation and strategy backtesting;
  a simple backtesting layer should be added later for validation.
