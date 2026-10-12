# Logistic Regression vs XGBoost

I wanted to test a simple question: does moving to a more complex model automatically give better results?

I used the Telco Customer Churn dataset and compared Logistic Regression with XGBoost under the same setup.

Both models used the same train/test split, preprocessing, and evaluation metrics.

## Setup

The dataset contains 7,043 customer records, with about 26.5% of the customers having churned.

For the comparison, I used:

- 80/20 stratified train/test split
- Standard scaling for numerical features
- One-hot encoding for categorical features
- Accuracy, Precision, Recall, F1 and ROC-AUC for evaluation

## Test set results

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 0.806 | 0.785 |
| Precision | 0.657 | 0.606 |
| Recall | 0.559 | 0.543 |
| F1 | 0.604 | 0.573 |
| ROC-AUC | 0.842 | 0.825 |

Logistic Regression performed better across all five metrics on the test set.

But one train/test split isn't enough to draw much of a conclusion, so I also ran 5-fold cross-validation.

## Cross-validation results

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 0.802 ± 0.012 | 0.780 ± 0.012 |
| Precision | 0.653 ± 0.026 | 0.602 ± 0.028 |
| Recall | 0.542 ± 0.041 | 0.508 ± 0.023 |
| F1 | 0.592 ± 0.030 | 0.551 ± 0.024 |
| ROC-AUC | 0.846 ± 0.013 | 0.822 ± 0.009 |

The same overall pattern showed up across the five folds.

## What I took away from this

I expected XGBoost to have an advantage because it can capture more complex relationships in the data. That wasn't the case with this setup.

That doesn't mean Logistic Regression is better than XGBoost. I haven't done serious hyperparameter tuning on XGBoost here, and another dataset or setup could easily produce a different result.

What this experiment reinforced for me is something much more practical:

**Start with a solid baseline, measure it properly, and make the more complex model prove that it's actually adding value.**

Sometimes it will. Sometimes it won't.

The full experiment, including preprocessing, model training, evaluation and cross-validation, is available in [`experiment.ipynb`](./experiment.ipynb).

## Running the experiment

The dataset itself isn't included in this repository.

This experiment uses the IBM Telco Customer Churn dataset. Download `Telco-Customer-Churn.csv` and place it inside the project's `data` directory:

```text
Churn-prediction/
├── data/
│   └── Telco-Customer-Churn.csv
├── experiments/
│   └── logistic-vs-xgboost/
│       ├── experiment.ipynb
│       └── README.md
└── requirements.txt