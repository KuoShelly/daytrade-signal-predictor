### 5. Stacking Ensemble Architecture

The ensemble combines three base classifiers with complementary modeling characteristics and a Logistic Regression meta-model.

| Model | Role |
|---|---|
| Logistic Regression (Base Model) | Provides a linear classification baseline and interpretable probability estimates. |
| Random Forest | Captures nonlinear relationships and feature interactions through an ensemble of decision trees. |
| LightGBM | Learns complex feature patterns using gradient-boosted decision trees. |
| Logistic Regression (Meta-model) | Learns how to combine the predictions of the three base models into a final ranking score. |

#### 5.1 Base Model Training

Logistic Regression, Random Forest, and LightGBM are trained to predict whether a stock belongs to the target group of high day-trading activity stocks.

The three base models generate prediction scores or probabilities for candidate stocks. These outputs are then used as input features for the Logistic Regression meta-model.

#### 5.2 Time-Series Out-of-Fold (OOF) Predictions

To reduce overfitting and prevent the meta-model from learning from overly optimistic in-sample predictions, time-series Out-of-Fold (OOF) predictions can be used to construct the meta-model's training dataset.

The process is as follows:

1. Train each base model using historical data from an earlier period.
2. Generate predictions for a subsequent period that was not used to train that model.
3. Repeat the process using progressively later periods while preserving chronological order.
4. Combine the OOF predictions from Logistic Regression, Random Forest, and LightGBM into a meta-feature matrix.
5. Train the Logistic Regression meta-model using the OOF predictions and their corresponding target labels.

This approach preserves the temporal structure of financial data and helps prevent future information from leaking into model training.

### 6. Evaluation Metric: Average Daily Top-30 Coverage

The primary evaluation metric measures how many stocks in the actual top-30 day-trading list are captured by the model's predicted top-30 list.

For each trading day \(d\):

\[
\operatorname{Coverage}_d =
\frac{|P_{30,d}\cap A_{30,d}|}{30}
\times 100\%
\]

Where:

- \(P_{30,d}\) is the set of the 30 stocks with the highest predicted scores on day \(d\).
- \(A_{30,d}\) is the set of the 30 actual target stocks based on day-trading activity on day \(d\).
- \(|P_{30,d}\cap A_{30,d}|\) is the number of stocks appearing in both lists.

The average daily coverage across \(D\) trading days is:

\[
\operatorname{AvgCoverage}
=\frac{1}{D}\sum_{d=1}^{D}\operatorname{Coverage}_d
\]

A coverage of 85% means that the model captures, on average, approximately 25–26 of the 30 actual target stocks in its daily predicted top-30 list.

### 7. Historical Experimental Results

In previous experiments using real Taiwanese stock market data, the Stacking Ensemble achieved approximately **85% average daily Top-30 Coverage**.

This metric was calculated by comparing the model's 30 highest-ranked stock predictions with the 30 actual target stocks identified by observed day-trading activity for each trading day.

The reported coverage represents the average proportion of actual target stocks captured by the predicted top-30 list. It is not classification accuracy or precision.

The original real-market data and complete experimental environment are not available in the current public implementation. The public version uses mock data to demonstrate the data-processing and modeling workflow, so the historical result cannot be independently reproduced from the mock data alone.
