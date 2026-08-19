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

## 🧪 Methodology

### Data Preprocessing
- Concatenate `Reviews Title` + `Reviews Text` into a single `text` column.
- Minimal text normalization: lowercasing, HTML tag removal, whitespace normalization.
- Drop rows with missing/invalid target (`Reviews Rating`).
- **Final sample size:** 2,445 reviews.

### Data Split
- **70% Train / 15% Validation / 15% Test**
- Stratified split to preserve class distribution across splits.
- Random seed: `42`

### Models Compared

| Model | Type | Description |
|-------|------|-------------|
| DummyRegressor (mean) | Baseline | Always predicts the training mean |
| TF-IDF + Ridge | Classical | TF-IDF vectorization (ngram 1–2, min_df=2) + Ridge Regression with GridSearchCV |
| DistilBERT (fine-tuned) | Neural | Hugging Face `distilbert-base-uncased` fine-tuned for regression (1 output neuron) |

### Hyperparameter Tuning
- **Ridge:** GridSearchCV over `alpha ∈ {0.1, 1.0, 10.0}` and `max_features ∈ {10000, 20000}`, 5-fold CV, scored by negative MAE.
- **DistilBERT:** Fixed hyperparameters (learning_rate=2e-5, epochs=2, batch_size=8, weight_decay=0.01). Best model selected on validation MAE.

### Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

All metrics are computed **once** on the held-out test set for fair comparison.

---

## 🚀 How to Run

### Prerequisites
- pip install numpy pandas scikit-learn torch transformers datasets matplotlib joblib
- numpy
- pandas
- scikit-learn
- torch
- transformers
- datasets
- matplotlib
- joblib
- ipywidgets
- pip install numpy pandas scikit-learn torch transformers datasets matplotlib joblib

### Reproducing Results 

This repository does not include trained model weights due to their size.
To regenerate all models and results:

1. Run all cells in `ML-project.ipynb`
2. Training takes ~8 minutes on CPU (DistilBERT, 2 epochs)
3. Models and metrics will be saved to the `results/` folder

Models generated:
- `results/ridge_model.joblib` (~small)
- `results/distilbert_model/` (~260MB)

## Authors
FLORE FAILA NGOY ZOLA
