# How Classification Thresholds Change Model Decisions

A classification model gives us probabilities, but we still need to decide when a probability should become a positive prediction.

I used the same Logistic Regression churn model and tested different classification thresholds without changing or retraining the model.

## Setup

I used the Telco Customer Churn dataset with the same preprocessing and train/test split used in the earlier experiment.

I tested these thresholds:

- 0.3
- 0.4
- 0.5
- 0.6
- 0.7

For each threshold, I compared accuracy, precision, recall, and F1.

## Results

| Threshold | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0.3 | 0.749 | 0.519 | 0.754 | 0.615 |
| 0.4 | 0.778 | 0.569 | 0.668 | 0.615 |
| 0.5 | 0.806 | 0.657 | 0.559 | 0.604 |
| 0.6 | 0.799 | 0.718 | 0.401 | 0.515 |
| 0.7 | 0.766 | 0.739 | 0.182 | 0.292 |

As the threshold increased, precision improved but recall dropped.

## Looking at actual customers

The difference became clearer when I looked at the actual prediction counts.

| Threshold | True Positives | False Positives | True Negatives | False Negatives |
|---:|---:|---:|---:|---:|
| 0.3 | 282 | 261 | 774 | 92 |
| 0.4 | 250 | 189 | 846 | 124 |
| 0.5 | 209 | 109 | 926 | 165 |
| 0.6 | 150 | 59 | 976 | 224 |
| 0.7 | 68 | 24 | 1011 | 306 |

At the default 0.5 threshold, the model caught 209 churners and missed 165.

At 0.3, it caught 282 churners and missed only 92.

That's 73 more churners caught, but it also incorrectly flagged 152 more non-churners.

## What I took from this

0.3 isn't necessarily better than 0.5. It just makes a different tradeoff.

If missing a churner is more costly, a lower threshold might make sense. If unnecessary retention actions are expensive, we may want to be more selective.

The model gives us the probability. How we use that probability should depend on the actual problem.

The full experiment is available in [`experiment.ipynb`](./experiment.ipynb).