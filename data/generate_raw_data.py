"""
generate_raw_data.py
---------------------
Generates a synthetic, intentionally MESSY e-commerce customer dataset.

This simulates the kind of raw data a data pipeline would receive in the
real world: missing values, inconsistent text casing, duplicate rows,
mixed date formats, and outliers. The ETL pipeline (etl_pipeline.py) is
built to clean and transform this into an analysis-ready dataset.

Run this once to produce data/raw_customers.csv, which etl_pipeline.py
then consumes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

genders = np.random.choice(["Male", "Female", "male", "FEMALE", None], size=N,
                            p=[0.35, 0.35, 0.1, 0.1, 0.1])

cities = np.random.choice(
    ["Mumbai", "delhi", "Bangalore", "BANGALORE", "Chennai", "Pune", None],
    size=N, p=[0.2, 0.2, 0.15, 0.1, 0.15, 0.15, 0.05]
)

signup_dates = pd.date_range("2022-01-01", "2025-12-31", periods=N)
# Mess up date formatting for a chunk of rows to simulate inconsistent sources
signup_dates = [
    d.strftime("%Y-%m-%d") if i % 3 != 0 else d.strftime("%d/%m/%Y")
    for i, d in enumerate(signup_dates)
]

age = np.random.normal(35, 12, N).round().astype(float)
age[np.random.choice(N, 40, replace=False)] = np.nan   # missing ages
age[np.random.choice(N, 5, replace=False)] = 150        # outliers/bad entries

annual_income = np.random.normal(600000, 200000, N).round(2)
annual_income[np.random.choice(N, 60, replace=False)] = np.nan

spending_score = np.random.randint(1, 100, N).astype(float)
spending_score[np.random.choice(N, 25, replace=False)] = np.nan

num_purchases = np.random.poisson(8, N)

df = pd.DataFrame({
    "customer_id": range(1001, 1001 + N),
    "gender": genders,
    "city": cities,
    "signup_date": signup_dates,
    "age": age,
    "annual_income": annual_income,
    "spending_score": spending_score,
    "num_purchases": num_purchases,
})

# Inject duplicate rows (common in real pipelines from repeated ingestion)
dupes = df.sample(15, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# Shuffle rows
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/raw_customers.csv", index=False)
print(f"raw_customers.csv generated with {len(df)} rows (including duplicates/missing/dirty values).")
