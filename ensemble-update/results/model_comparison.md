# Model Comparison Results (Mock data)

> **Mock / synthetic data only.** These numbers come from randomly generated data and are
> NOT real market results. They only verify that the comparison pipeline works.


## Pure-noise Mock data (leakage check)

- Train period: 20231225~20241016, validation period: 20241017~20241231
- Train rows: 10,155, validation rows: 2,586, validation positive ratio: 59.9%
- Seeds: [0, 1, 2, 3, 4] (mean ± std across seeds)

| Model                | AUC-ROC       | Top-30 Hit Rate   | Accuracy      | Precision     | Recall        | F1            | Daily Hit Rate   |   Fit Seconds |
|:---------------------|:--------------|:------------------|:--------------|:--------------|:--------------|:--------------|:-----------------|--------------:|
| Baseline (當沖比例_lag1) | 0.497 ± 0.009 | 0.595 ± 0.007     | —             | —             | —             | —             | —                |           0   |
| Logistic Regression  | 0.515 ± 0.012 | 0.600 ± 0.006     | 0.512 ± 0.010 | 0.609 ± 0.009 | 0.514 ± 0.012 | 0.558 ± 0.011 | 0.514 ± 0.012    |           0.1 |
| Random Forest        | 0.508 ± 0.010 | 0.603 ± 0.005     | 0.525 ± 0.011 | 0.605 ± 0.009 | 0.595 ± 0.025 | 0.599 ± 0.015 | 0.595 ± 0.025    |           5.8 |
| LightGBM             | 0.500 ± 0.009 | 0.597 ± 0.007     | 0.599 ± 0.001 | 0.599 ± 0.001 | 1.000 ± 0.000 | 0.749 ± 0.001 | 1.000 ± 0.000    |           0.5 |
| Ensemble: Average    | 0.514 ± 0.012 | 0.601 ± 0.006     | 0.590 ± 0.003 | 0.599 ± 0.002 | 0.949 ± 0.017 | 0.735 ± 0.005 | 0.949 ± 0.017    |           6.3 |
| Ensemble: Stacking   | 0.507 ± 0.009 | 0.604 ± 0.006     | 0.508 ± 0.029 | 0.606 ± 0.007 | 0.508 ± 0.135 | 0.544 ± 0.070 | 0.508 ± 0.136    |          19.2 |


## Synthetic data with a planted signal

- Train period: 20231225~20241016, validation period: 20241017~20241231
- Train rows: 20,480, validation rows: 5,215, validation positive ratio: 31.2%
- Seeds: [0, 1, 2, 3, 4] (mean ± std across seeds)

| Model                | AUC-ROC       | Top-30 Hit Rate   | Accuracy      | Precision     | Recall        | F1            | Daily Hit Rate   |   Fit Seconds |
|:---------------------|:--------------|:------------------|:--------------|:--------------|:--------------|:--------------|:-----------------|--------------:|
| Baseline (當沖比例_lag1) | 0.895 ± 0.010 | 0.725 ± 0.015     | —             | —             | —             | —             | —                |           0   |
| Logistic Regression  | 0.903 ± 0.011 | 0.736 ± 0.018     | 0.815 ± 0.012 | 0.662 ± 0.016 | 0.833 ± 0.013 | 0.738 ± 0.014 | 0.833 ± 0.013    |           0.3 |
| Random Forest        | 0.901 ± 0.010 | 0.733 ± 0.016     | 0.812 ± 0.011 | 0.655 ± 0.015 | 0.840 ± 0.011 | 0.736 ± 0.012 | 0.840 ± 0.011    |          10.7 |
| LightGBM             | 0.918 ± 0.010 | 0.756 ± 0.017     | 0.841 ± 0.012 | 0.715 ± 0.020 | 0.816 ± 0.015 | 0.762 ± 0.016 | 0.816 ± 0.015    |           3.9 |
| Ensemble: Average    | 0.914 ± 0.010 | 0.751 ± 0.019     | 0.831 ± 0.011 | 0.690 ± 0.017 | 0.836 ± 0.013 | 0.756 ± 0.014 | 0.836 ± 0.013    |          15.1 |
| Ensemble: Stacking   | 0.918 ± 0.010 | 0.758 ± 0.017     | 0.831 ± 0.012 | 0.685 ± 0.018 | 0.850 ± 0.014 | 0.759 ± 0.013 | 0.850 ± 0.014    |          44.6 |
