# Machine-Learning-project
Predicting Ratings from Text: A comparative Regression Experiment
# Wine Review Rating Prediction (Regression)

## Overview
This project predicts the numerical review rating (1–5) from review text.
It contains a methodologically correct comparative experiment:
- Classical model: TF-IDF + Ridge Regression
- Neural model: DistilBERT fine-tuned for regression

This structure follows the course project requirements (preprocessing, comparative experiment, documented code, short paper). 

## Dataset
Input file: `wine reviews.csv` (provided).

Target:
- `Reviews Rating` (numeric)

Main features:
- `Reviews Title`, `Reviews Text` → concatenated into a single `text` field
Optional metadata:
- `Reviews do Recommend`, `Reviews Num Helpful`

Preprocessed data: data/processed/ (train/val/test)

## Methodology
We use a 70/15/15 train/validation/test split (seed=42):
- Train: fit model parameters
- Validation: select hyperparameters / best epoch
- Test: final evaluation once

Metrics:
- MAE, RMSE, R²

## How to run
1. Install requirements
2. Run notebook or scripts

## Results
See results/metrics.json

## Authors
FLORE FAILA NGOY ZOLA
