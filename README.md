Day-Trade Signal Predictor

A Stacking Ensemble for Next-Day Taiwanese Stock Day-Trading Prediction

A machine learning project designed to identify Taiwanese stocks with high next-day day-trading activity using multi-source market features and a three-model ensemble.

In previous experiments using real market data, the model achieved approximately 85% average daily coverage of the actual top-30 target stocks. The current public version uses mock data to demonstrate the data-processing and modeling workflow. Because the original real-market data is not publicly available, the historical result cannot be independently reproduced using the mock-data implementation.

1. Project Overview

Objectives

Integrate multi-source Taiwanese stock market data into a consistent feature engineering pipeline.

Predict stocks likely to exhibit high day-trading activity on the following trading day.

Combine Logistic Regression, Random Forest, and LightGBM through a Stacking Ensemble.

Address class imbalance to improve the identification of target stocks.

Evaluate model performance using time-aware validation and daily top-30 coverage.

Tech Stack

Language: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn, LightGBM

Model Architecture: Stacking Ensemble

Evaluation: Recall, Precision, F1-score, ROC-AUC, Average Daily Top-30 Coverage

2. Data Sources and Target Definition

Data Sources

The original pipeline integrates multiple Taiwanese stock market data sources, including:

Intraday technical indicators

Trading volume and market rankings

Moving averages

Margin trading and short-selling data

Day-trading volume and day-trading ratios

To support public demonstration without access to the original market data infrastructure, the project uses a data-source abstraction layer and a MockMarketDataSource implementation.

The mock data is intended to demonstrate pipeline functionality, not to represent real market conditions or validate predictive performance.

Target Label Construction

The prediction target represents stocks with high next-day day-trading activity.

Exclude stocks with daily trading volume below 10,000 shares, according to the project's eligibility criteria.

Calculate the ratio of day-trading volume to total trading volume.

Rank eligible stocks by their day-trading ratios.

Label the top 30 eligible stocks as the positive class and the remaining eligible stocks as the negative class.

Labels must be constructed using information available for the corresponding prediction date, without incorporating future information into the features.

3. Feature Engineering

The feature engineering pipeline transforms market data into structured inputs for supervised learning.

Lagged Features: Generate historical features from previous trading days to capture changes in trading activity and market behavior.

Derived Features: Construct additional indicators from available market variables.

Multi-source Integration: Align data from different sources by stock identifier and trading date.

Missing-value Handling: Apply consistent missing-value treatment based on feature characteristics.

Leakage Prevention: Ensure that features use only information available at the prediction time.

4. Class Imbalance Handling

The positive class represents a subset of eligible stocks, which may cause a classifier to favor the negative class and miss relevant target stocks.

Class imbalance should therefore be addressed during model training and evaluated using chronological validation.

4.1 Class Weighting

Class weighting assigns different loss weights to different classes, increasing the penalty associated with misclassifying positive samples.

Potential implementations include:

Logistic Regression: class_weight="balanced" or explicitly defined class weights.

Random Forest: class_weight="balanced" or class_weight="balanced_subsample".

LightGBM: scale_pos_weight or an appropriate class_weight configuration.

Weight settings should be selected using training data and validated on later periods. They do not need to be identical across all three models.

4.2 Ranking and Threshold Selection

Because the final objective is to generate a daily top-30 candidate list, model evaluation should focus on ranking quality as well as conventional classification metrics.

Probability thresholds and ranking strategies should be selected using historical validation data and then evaluated on a separate, later test period.

4.3 Optional Oversampling

SMOTE or other oversampling techniques can be investigated if class weighting alone does not adequately improve minority-class detection.

Oversampling must be applied only to the training portion of each validation split. Synthetic samples should not be generated before splitting the complete dataset, as this can introduce data leakage.

For financial market features, synthetic samples should also be examined for plausibility and consistency with the underlying feature relationships.

Implementation note: These are available strategies for handling class imbalance. The exact weighting parameters and whether oversampling was used should be documented according to the corresponding implementation.
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
