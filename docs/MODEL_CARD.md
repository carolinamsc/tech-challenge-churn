# Model Card — Churn Prediction

> **Status:** Draft — final performance values will be filled after the reproducible training run.

## 1. Model overview

This project predicts whether a telecom customer is likely to churn. The prediction is intended to support customer-retention prioritization, not to make an irreversible automated decision.

The candidate models are:

- Logistic Regression — required baseline;
- Random Forest;
- MLPClassifier (Scikit-Learn).

All candidates share the same preprocessing pipeline and are compared using stratified cross-validation.

## 2. Intended use

The model is intended to rank customers by estimated churn risk so a retention team can prioritize outreach.

It should not be used as the sole basis for denying service, changing contractual terms, or making decisions about a customer without human review.

## 3. Data

The project uses the public IBM Telco Customer Churn dataset. The raw dataset contains 7,043 customer records and 21 columns, with `Churn` as the target.

`customerID` is treated as an identifier and excluded from model features. `TotalCharges` is converted to numeric and missing values are handled inside the preprocessing pipeline.

## 4. Training and evaluation

- Random seed: `42`
- Test split: `20%`
- Cross-validation: stratified 5-fold
- Candidate metrics: ROC-AUC, F1, precision, recall and accuracy
- The test set is kept untouched until model selection is complete.

## 5. Threshold

The default classification threshold is `0.50`, but the project includes threshold analysis. The final threshold should be selected using validation results and the relative business cost of false negatives versus false positives.

## 6. Performance

Final CV and holdout metrics are intentionally not hard-coded before execution. This prevents undocumented or fabricated results from entering the deliverable.

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | pending | pending | pending | pending |
| Random Forest | pending | pending | pending | pending |
| MLPClassifier | pending | pending | pending | pending |

## 7. Limitations and risks

- The dataset represents a particular telecom customer population and may not generalize to other businesses.
- Historical churn patterns can encode existing commercial biases.
- A probability estimate is not a causal explanation for why a customer will churn.
- Customer behavior and pricing can change over time, causing distribution drift.
- The model should be monitored and periodically revalidated if deployed.

## 8. Reproducibility

The project centralizes the random seed and split configuration. Preprocessing is part of the Scikit-Learn pipeline so training transformations are learned without using the held-out test set.

Run the training and comparison commands from the repository root after installing the project dependencies.
