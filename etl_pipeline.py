"""
etl_pipeline.py
================
CodTech Internship — Data Science Task 1: Data Pipeline Development

Builds an automated ETL (Extract, Transform, Load) pipeline that:
  1. EXTRACT   -> Reads raw, messy customer data from a CSV file.
  2. TRANSFORM -> Cleans, standardizes, engineers features, and preprocesses
                  the data using pandas + a scikit-learn Pipeline
                  (imputation, scaling, one-hot encoding).
  3. LOAD      -> Writes the cleaned dataset and the fitted preprocessing
                  pipeline to the output/ folder, ready for downstream
                  modeling or analysis.

Usage:
    python etl_pipeline.py

Requirements:
    pandas, numpy, scikit-learn, joblib  (see requirements.txt)
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RAW_DATA_PATH = "data/raw_customers.csv"
CLEAN_DATA_PATH = "output/clean_customers.csv"
PROCESSED_ARRAY_PATH = "output/processed_features.csv"
PIPELINE_PATH = "output/preprocessing_pipeline.joblib"

NUMERIC_FEATURES = ["age", "annual_income", "spending_score", "num_purchases"]
CATEGORICAL_FEATURES = ["gender", "city"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("etl")


# --------------------------------------------------------------------------- #
# 1. EXTRACT
# --------------------------------------------------------------------------- #
def extract(path: str) -> pd.DataFrame:
    """Load raw data from a CSV source."""
    logger.info(f"Extracting raw data from '{path}'...")
    df = pd.read_csv(path)
    logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns.")
    return df


# --------------------------------------------------------------------------- #
# 2. TRANSFORM
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row/column-level cleaning done in pandas BEFORE the sklearn pipeline:
    - Drop exact duplicate rows
    - Standardize text casing in categorical columns
    - Parse inconsistent date formats
    - Remove impossible/outlier values
    - Engineer a couple of extra features
    """
    df = df.copy()
    before = len(df)

    # Drop duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicate rows.")

    # Standardize categorical text (fixes 'male' vs 'Male' vs 'MALE')
    df["gender"] = df["gender"].str.strip().str.capitalize()
    df["city"] = df["city"].str.strip().str.title()

    # Parse mixed date formats (YYYY-MM-DD and DD/MM/YYYY) into one format
    df["signup_date"] = pd.to_datetime(df["signup_date"], format="mixed", dayfirst=True, errors="coerce")

    # Remove impossible values (data entry errors), let imputer handle later
    df.loc[(df["age"] < 10) | (df["age"] > 100), "age"] = np.nan

    # Feature engineering: how many years since signup (useful for modeling)
    reference_date = pd.Timestamp("2026-07-09")
    df["tenure_years"] = ((reference_date - df["signup_date"]).dt.days / 365.25).round(2)

    logger.info("Cleaning complete: standardized text, fixed dates, removed outliers.")
    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
    - Imputes missing numeric values with the median, then scales them
    - Imputes missing categorical values with the most frequent value,
      then one-hot encodes them
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def transform(df: pd.DataFrame):
    """Apply pandas cleaning, then fit/apply the sklearn preprocessing pipeline."""
    clean_df = clean_data(df)

    preprocessor = build_preprocessing_pipeline()
    logger.info("Fitting scikit-learn preprocessing pipeline (impute + scale + encode)...")
    processed_array = preprocessor.fit_transform(clean_df)

    # Recover readable column names after one-hot encoding
    cat_columns = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_columns = NUMERIC_FEATURES + list(cat_columns)

    processed_df = pd.DataFrame(processed_array, columns=all_columns)
    logger.info(f"Transformation complete: {processed_df.shape[1]} features after encoding.")

    return clean_df, processed_df, preprocessor


# --------------------------------------------------------------------------- #
# 3. LOAD
# --------------------------------------------------------------------------- #
def load(clean_df: pd.DataFrame, processed_df: pd.DataFrame, preprocessor: ColumnTransformer):
    """Persist the cleaned dataset, the fully processed feature matrix, and the fitted pipeline."""
    os.makedirs("output", exist_ok=True)

    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    logger.info(f"Loaded human-readable cleaned dataset -> '{CLEAN_DATA_PATH}'")

    processed_df.to_csv(PROCESSED_ARRAY_PATH, index=False)
    logger.info(f"Loaded model-ready processed features -> '{PROCESSED_ARRAY_PATH}'")

    joblib.dump(preprocessor, PIPELINE_PATH)
    logger.info(f"Saved fitted preprocessing pipeline -> '{PIPELINE_PATH}' (reusable on new data)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_pipeline():
    logger.info("===== ETL PIPELINE START =====")
    raw_df = extract(RAW_DATA_PATH)
    clean_df, processed_df, preprocessor = transform(raw_df)
    load(clean_df, processed_df, preprocessor)
    logger.info("===== ETL PIPELINE COMPLETE =====")

    # Quick summary printed to console
    print("\nSummary:")
    print(f"  Raw rows:            {len(raw_df)}")
    print(f"  Clean rows:          {len(clean_df)}")
    print(f"  Missing values left: {clean_df[NUMERIC_FEATURES].isna().sum().sum()} (numeric, pre-impute)")
    print(f"  Final feature count: {processed_df.shape[1]}")


if __name__ == "__main__":
    run_pipeline()
