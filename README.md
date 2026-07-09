# Customer Data ETL Pipeline

## Overview
This project implements an automated **ETL (Extract, Transform, Load)** pipeline
for a raw, messy e-commerce customer dataset, using **pandas** for data cleaning
and **scikit-learn** for preprocessing (imputation, scaling, encoding).

## Problem
Real-world data is rarely clean. This pipeline handles common issues:
- Missing values (age, income, spending score)
- Inconsistent text casing (`Male` / `male` / `MALE`)
- Mixed date formats (`YYYY-MM-DD` and `DD/MM/YYYY`)
- Duplicate rows
- Outliers / impossible values (e.g. age = 150)

## Pipeline Stages
| Stage | What happens | Tools |
|---|---|---|
| **Extract** | Load raw CSV | `pandas.read_csv` |
| **Transform** | Deduplicate, standardize text, parse dates, remove outliers, engineer features, impute missing values, scale numeric features, one-hot encode categoricals | `pandas`, `sklearn.Pipeline`, `sklearn.ColumnTransformer` |
| **Load** | Save cleaned data, model-ready feature matrix, and the fitted pipeline object | `pandas.to_csv`, `joblib.dump` |

## Project Structure
```
data-pipeline-project/
├── data/
│   ├── generate_raw_data.py   # generates synthetic messy dataset
│   └── raw_customers.csv      # raw input (generated)
├── output/
│   ├── clean_customers.csv         # human-readable cleaned data
│   ├── processed_features.csv      # scaled/encoded, model-ready
│   └── preprocessing_pipeline.joblib  # reusable fitted pipeline
├── etl_pipeline.py             # main ETL script
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt

# 1. Generate the raw sample dataset (skip if you have your own raw data)
python data/generate_raw_data.py

# 2. Run the ETL pipeline
python etl_pipeline.py
```

Output files are written to `output/`.

## Key Design Choices
- **`ColumnTransformer`** keeps numeric and categorical preprocessing logic
  separate and composable, and is the standard scikit-learn approach for
  production pipelines.
- **Median imputation** for numeric columns is robust to the outliers present
  in the raw data.
- The fitted pipeline is **saved with `joblib`** so it can be reused on new,
  unseen data without refitting — important for production ETL where you
  transform new batches the same way each time.
- Logging is used throughout so each pipeline run produces a clear audit
  trail of what was cleaned and how many rows/values were affected.

## Results
The pipeline generates:

- Clean customer dataset
- Machine-learning ready feature matrix
- Reusable preprocessing pipeline

This project demonstrates a reusable ETL workflow for customer data preprocessing.

---

