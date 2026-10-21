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
| `src/model_comparison.py` | *(added after refactoring)* LightGBM vs. Logistic Regression vs. Random Forest under one shared split |
| `src/synthetic_data.py` | *(added after refactoring)* Mock data source with a deliberately planted, known signal |
| `compare_models.py` | *(added after refactoring)* CLI for the model comparison experiment |
| `tests/test_model_comparison.py` | *(added after refactoring)* Tests that the comparison itself is fair and correct |
| `src/ensemble.py` | *(added after refactoring)* Ensembling the three models: `AverageEnsemble` and time-aware `StackingEnsemble` |
| `src/ensemble_model.py` | *(added after refactoring)* `EnsembleDayTradeModel` — drop-in replacement for `DayTradeModel` in the main pipeline |
| `tests/test_ensemble.py` | *(added after refactoring)* Tests that stacking never trains on future data, plus end-to-end checks |

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
python train.py --model stacking --data synthetic   # three-model ensemble (Section 6), synthetic data
python compare_models.py            # model comparison experiment (Section 6), ~10 min
pytest tests/
```

`MockMarketDataSource` is used by default to generate demo data. To connect a real data source,
simply implement a new `MarketDataSource` subclass (see the example code at the bottom of `data_interface.py`)
and replace `MockMarketDataSource()` in `train.py`.
The rest of the pipeline requires no changes at all.

## 6. Model Comparison and Ensembling Experiment (Added After Refactoring, Mock Data Only)

> ⚠️ **Read this first.**
> - This experiment was **added after the refactoring of this public repo**. It was **not** part of the
>   original project work done on the company's internal data, and none of the numbers below come from that work.
> - It runs **only on Mock / synthetic data** generated in this repo. The numbers are **not real market
>   results** and say nothing about how these models would perform on real Taiwan stock data.
> - The original version (Section 4) trained LightGBM only; it was never compared against other models,
>   ensembled, or checked against a naive baseline on real data.

### Why add this experiment?
The original version chose LightGBM without checking whether a simpler model — or no model at all —
would do just as well. This experiment builds the machinery to answer that question fairly, so that the
same comparison can be run as soon as a real `MarketDataSource` is plugged in.

### Candidates
| Model | Setup |
|---|---|
| Baseline (`當沖比例_lag1`) | No training. Score = yesterday's day-trading ratio. Tests whether a model learns more than "hot yesterday, hot today." |
| Logistic Regression | `StandardScaler` + `class_weight="balanced"` |
| Random Forest | 300 trees, `max_depth=10`, `min_samples_leaf=50`, `class_weight="balanced"` |
| LightGBM | Same parameters as `src/config.py`, `scale_pos_weight = negatives / positives` |
| Ensemble: Average | Equal-weight mean of the three models' predicted probabilities |
| Ensemble: Stacking | Logistic Regression meta-model on the three models' logit probabilities, trained on **time-ordered** out-of-fold predictions (see below) |

### Fairness rules (`src/model_comparison.py`)
1. **Same features**: every model uses the same `LaggedFeatureBuilder` output.
2. **Same time-based split**: `time_based_split()` is called once; every model is evaluated on the exact same validation rows, all dated after the training period.
3. **The validation set never influences training**: the main `DayTradeModel` uses the validation set for LightGBM early stopping, which slightly favors LightGBM. In this experiment, early stopping uses only the last 10% of *training* dates.
4. **Consistent class-imbalance handling** across models (see table above).
5. **Threshold-free metrics are reported** alongside the threshold-0.5 metrics: AUC-ROC and the **Top-30 Hit Rate** (each day, take the 30 highest-scoring stocks and measure how many are real positives). Because the label itself is "top 30 per day," this is the metric closest to how the list would actually be used.
6. **Multiple seeds**: each scenario is regenerated with 5 random seeds; results are reported as mean ± std.

### Two data scenarios
- **Pure-noise Mock data** (`MockMarketDataSource`): every column is independent random noise, so nothing is predictable. This is a **leakage check** — if any model scored clearly above AUC 0.5, the features or split would be leaking future information.
- **Synthetic data with a planted signal** (`SyntheticSignalDataSource`): a known signal is deliberately injected — a per-stock day-trading tendency, day-to-day persistence (AR(1)), a **non-linear** effect (an extreme RSI(5) yesterday pushes today's day-trading ratio up), and amplitude correlated with the latent tendency. This checks whether models can find a signal that is *known* to exist. 100 stocks are generated so the positive ratio (~31%) is close to the original version's 25-32%.

### Results (Mock data — NOT real market results)
Period 2023/12/20 - 2024/12/31, 80/20 split by date (validation 2024/10/17 - 2024/12/31), 5 seeds, mean ± std.
Full output: `results/model_comparison.md`; per-seed details: `results/*_runs.csv`.

**Pure-noise Mock data (leakage check)** — 2,586 validation rows, positive ratio 59.9%

| Model | AUC-ROC | Top-30 Hit Rate | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Baseline (`當沖比例_lag1`) | 0.497 ± 0.009 | 0.595 ± 0.007 | — | — | — | — |
| Logistic Regression | 0.515 ± 0.012 | 0.600 ± 0.006 | 0.512 ± 0.010 | 0.609 ± 0.009 | 0.514 ± 0.012 | 0.558 ± 0.011 |
| Random Forest | 0.508 ± 0.010 | 0.603 ± 0.005 | 0.525 ± 0.011 | 0.605 ± 0.009 | 0.595 ± 0.025 | 0.599 ± 0.015 |
| LightGBM | 0.500 ± 0.009 | 0.597 ± 0.007 | 0.599 ± 0.001 | 0.599 ± 0.001 | 1.000 ± 0.000 | 0.749 ± 0.001 |
| Ensemble: Average | 0.514 ± 0.012 | 0.601 ± 0.006 | 0.590 ± 0.003 | 0.599 ± 0.002 | 0.949 ± 0.017 | 0.735 ± 0.005 |
| Ensemble: Stacking | 0.507 ± 0.009 | 0.604 ± 0.006 | 0.508 ± 0.029 | 0.606 ± 0.007 | 0.508 ± 0.135 | 0.544 ± 0.070 |

**Synthetic data with a planted signal** — 5,215 validation rows, positive ratio 31.2%

| Model | AUC-ROC | Top-30 Hit Rate | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Baseline (`當沖比例_lag1`) | 0.895 ± 0.010 | 0.725 ± 0.015 | — | — | — | — |
| Logistic Regression | 0.903 ± 0.011 | 0.736 ± 0.018 | 0.815 ± 0.012 | 0.662 ± 0.016 | 0.833 ± 0.013 | 0.738 ± 0.014 |
| Random Forest | 0.901 ± 0.010 | 0.733 ± 0.016 | 0.812 ± 0.011 | 0.655 ± 0.015 | 0.840 ± 0.011 | 0.736 ± 0.012 |
| LightGBM | 0.918 ± 0.010 | 0.756 ± 0.017 | 0.841 ± 0.012 | 0.715 ± 0.020 | 0.816 ± 0.015 | 0.762 ± 0.016 |
| Ensemble: Average | 0.914 ± 0.010 | 0.751 ± 0.019 | 0.831 ± 0.011 | 0.690 ± 0.017 | 0.836 ± 0.013 | 0.756 ± 0.014 |
| Ensemble: Stacking | 0.918 ± 0.010 | 0.758 ± 0.017 | 0.831 ± 0.012 | 0.685 ± 0.018 | 0.850 ± 0.014 | 0.759 ± 0.013 |

Average training time per run on the planted-signal data: LightGBM 3.9 s, Logistic Regression 0.3 s, Random Forest 10.7 s, Average 15.1 s, Stacking 44.6 s.

(The baseline has no probability output, so threshold-based metrics do not apply. "Daily Hit Rate" from Section 4 is also in the CSV output; under its definition it equals recall averaged per day.)

### What these results do and do not show
- **The pipeline passes the leakage check.** On pure noise, every model's per-seed AUC falls between 0.486 and 0.526 and the Top-30 Hit Rate equals the positive ratio (~0.60) — i.e. random guessing. With ~2,600 validation rows, AUC differences of this size are within chance.
- **Threshold metrics can look good on pure noise.** On noise data, LightGBM's early stopping keeps only one or two trees and the model predicts "positive" for every row, giving Recall 1.000 and F1 0.749 with no predictive ability at all. This is why AUC and the Top-30 Hit Rate are the primary metrics here.
- **The naive baseline is a strong bar.** On the planted-signal data, yesterday's day-trading ratio alone reaches AUC 0.895; the trained models add only +0.006 (Random Forest) to +0.023 (LightGBM) on average. In the original real-data experiment, `當沖比例_lag1` was also the top feature (Section 4), so **comparing against this baseline is the most important next step on real data**.
- **Among the single models, LightGBM came out ahead on the planted-signal data**, and the gap is consistent: within the same seed, its AUC exceeded Logistic Regression's by +0.011 to +0.018 and Random Forest's by +0.014 to +0.020 in all 5 seeds. However, **this is expected by construction**: the generator includes a non-linear threshold effect (extreme RSI), which tree models can capture and a linear model cannot. It reflects how the synthetic data was designed, not anything about real markets.
- **Neither ensemble clearly beat LightGBM alone** on the planted-signal data (details in the next subsection).
- **No conclusion about real-world model choice can be drawn from this experiment.** It only shows that the comparison code works and that the models behave sensibly on data where the answer is known.

### Ensembling: Average vs. Stacking — which one and why

**Chosen method: time-aware Stacking** (`StackingEnsemble`, `src/ensemble.py`), with Average kept as a simpler reference.

**How the stacking avoids look-ahead bias.** Standard stacking trains the meta-model on out-of-fold predictions
from a *random* K-fold. On time-series data that lets base models predict the past using the future, so their
out-of-fold scores look better than they will in production, and the meta-model learns the wrong weights.
Here the training period is split by date into 5 consecutive blocks; for fold *k* the base models are trained on
blocks 0..k-1 and predict block *k* (an expanding window). Every out-of-fold prediction therefore uses only
earlier data, exactly as in live use. The base models are then retrained on the full training period to score the
validation set. `tests/test_ensemble.py` checks that no fold ever trains on a date later than the one it predicts.

**Why Stacking over Average** (based on the Mock-data results above):
- **Average is dragged down by weaker members.** Equal weights give Logistic Regression and Random Forest the same
  say as LightGBM. On the planted-signal data, Average's AUC was *below* LightGBM alone in all 5 seeds
  (−0.002 to −0.006).
- **Stacking learns the weights.** Its meta-model put most of the weight on LightGBM (coefficient 0.66 to 0.84),
  some on Logistic Regression (0.15 to 0.24) and almost none on Random Forest (−0.08 to 0.05) — it learned to
  ignore the redundant tree model. Its AUC matched LightGBM within ±0.002 in every seed, and the weights are
  directly interpretable (printed by `train.py --model stacking`).
- **Stacking adapts when member strengths change.** On real data the relative strength of the three models is
  unknown; with fixed equal weights a weak member always costs accuracy, while stacking can re-weight after retraining.

**Costs and risks of Stacking (to be honest about):**
- **It did not outperform LightGBM alone** on this synthetic data — ensembling only helps when members make
  *different* errors, and here LightGBM already captured most of the planted signal.
- **About 3× the training time of Average** (44.6 s vs. 15.1 s) because base models are trained once per fold plus a final refit.
- **The meta-model can fit noise when there is no signal.** On pure-noise data its weights were unstable across
  seeds (LightGBM's coefficient ranged from −0.90 to +0.56) and its recall swung widely (0.508 ± 0.135).
  On real data, the meta-weights should be checked for stability across retraining windows before relying on them.
- The first time block never gets an out-of-fold prediction, so the meta-model is trained on ~80% of the training period.

When training data is short, or the meta-weights are unstable across retraining windows, **Average is the safer fallback**.

Use it in the main pipeline:
```bash
python train.py --model stacking --data synthetic   # or --model average; default is --model lightgbm
```

### Reproduce
```bash
python compare_models.py                                 # both scenarios, seeds 0-4 (~10 min)
python compare_models.py --scenario signal --seeds 0 1 2 # one scenario, custom seeds
```
To run the same comparison on real data, implement a `MarketDataSource` and pass it to
`run_model_comparison()` in `src/model_comparison.py`.

## 7. Known Limitations and Future Directions

- Currently uses a single time-window train/validation split; walk-forward
  cross-validation has not been done yet, so model stability has only been verified on one validation window.
- The model comparison and ensembling (Section 6) have only been run on Mock / synthetic data. Whether LightGBM,
  the stacking ensemble, or even the `當沖比例_lag1` baseline is best on real data is still an open question.
- The stacking ensemble's members use fixed hyperparameters; they have not been tuned, and the meta-model's
  weight stability across retraining windows has not been tested.
- The main `DayTradeModel` sets `scale_pos_weight = 1 / positive_ratio` (carried over from the original version),
  while the comparison experiment uses the more common `negatives / positives`. The two are close at a ~30% positive
  ratio but diverge as the ratio changes; this has not been unified yet.
- Feature importance shows the model relies heavily on `_lag1` features, meaning that when market conditions change quickly,
  the model may need more frequent retraining; "how often to retrain" has not yet been quantified.
- There is currently no simulation of transaction costs, slippage, or practical executability (liquidity impact).
  High prediction accuracy does not directly mean a profitable strategy. This is a common gap between model evaluation and strategy backtesting;
  a simple backtesting layer should be added later for validation.
