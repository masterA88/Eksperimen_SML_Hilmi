"""Automated preprocessing pipeline for the Telco Customer Churn dataset.

This module is the automation counterpart of the manual exploration done in
``Eksperimen_Hilmi.ipynb``. It performs exactly the same steps (loading,
cleaning, encoding, scaling, splitting) but wrapped into reusable functions so
the preprocessing can be reproduced with a single call and returns data that is
ready to be trained on.

Author: Hilmi (https://master-hilmi.vercel.app/)
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns that describe the customer but carry no predictive signal.
ID_COLUMNS = ["customerID"]
TARGET_COLUMN = "Churn"
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV into a DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame: drop IDs, fix TotalCharges, drop missing & duplicates.

    ``TotalCharges`` is stored as text and contains blank strings for brand-new
    customers (tenure == 0). We coerce it to numeric, which turns those blanks
    into NaN, then drop the few rows affected.
    """
    df = df.copy()

    # Drop identifier columns that should not enter the model.
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    # TotalCharges arrives as object because of blank strings -> coerce to float.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Handle missing values (the coerced blanks) and exact duplicates.
    df = df.dropna()
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target and all categorical features into numeric form.

    * Target ``Churn`` (Yes/No) -> 1/0.
    * ``SeniorCitizen`` is already 0/1.
    * Every remaining object column is one-hot encoded (drop_first to avoid the
      dummy-variable trap).
    """
    df = df.copy()

    # Encode the binary target.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    # One-hot encode the categorical predictors.
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Convert any boolean dummy columns to integers for a clean numeric frame.
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Split into train/test and standard-scale the numeric columns.

    The scaler is fit on the training data only (to avoid leakage) and applied
    to both splits. Returns the split frames plus the fitted scaler.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    numeric = [c for c in NUMERIC_COLUMNS if c in X_train.columns]
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric] = scaler.fit_transform(X_train[numeric])
    X_test[numeric] = scaler.transform(X_test[numeric])

    return X_train, X_test, y_train, y_test, scaler


def preprocess_data(
    path: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Run the full pipeline and return train/test ready-to-train data.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    df = load_data(path)
    df = clean_data(df)
    df = encode_features(df)
    return split_and_scale(df, test_size=test_size, random_state=random_state)


def save_outputs(X_train, X_test, y_train, y_test, scaler, output_dir: str) -> None:
    """Persist the processed splits (CSV) and the fitted scaler (joblib)."""
    os.makedirs(output_dir, exist_ok=True)

    train = X_train.copy()
    train[TARGET_COLUMN] = y_train.values
    test = X_test.copy()
    test[TARGET_COLUMN] = y_test.values

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the Telco Customer Churn dataset."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "telco_raw.csv"),
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "telco_preprocessing"),
        help="Directory where the processed dataset is written.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        args.input, test_size=args.test_size, random_state=args.random_state
    )
    save_outputs(X_train, X_test, y_train, y_test, scaler, args.output)

    print("Preprocessing complete.")
    print(f"  Train shape : {X_train.shape}")
    print(f"  Test shape  : {X_test.shape}")
    print(f"  Features    : {X_train.shape[1]}")
    print(f"  Churn rate  : {float(np.mean(y_train)):.3f} (train)")
    print(f"  Output dir  : {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
