import pandas as pd
import numpy as np


def _ensure_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Guarantees df[col] is a 1D pandas Series.
    Handles duplicate columns safely.
    """
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)

    data = df[col]

    # If duplicate columns -> DataFrame, pick first column
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    return pd.to_numeric(data, errors="coerce").fillna(0.0)


def clean_and_engineer_upi(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. HARD COPY + REMOVE DUPLICATE COLUMNS
    # ------------------------------------------------------------------
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # ------------------------------------------------------------------
    # 2. NORMALIZE COLUMN NAMES (CRITICAL)
    # ------------------------------------------------------------------
    rename_map = {
        "oldbalanceorg": "oldbalanceOrig",
        "oldbalanceOrg": "oldbalanceOrig",
        "newbalanceorg": "newbalanceOrig",
        "newbalanceOrg": "newbalanceOrig",
        "oldbalancedest": "oldbalanceDest",
        "newbalancedest": "newbalanceDest",
    }

    df.rename(columns=rename_map, inplace=True)

    # ------------------------------------------------------------------
    # 3. REQUIRED BASE COLUMNS
    # ------------------------------------------------------------------
    required_cols = [
        "step",
        "type",
        "amount",
        "oldbalanceOrig",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    # ------------------------------------------------------------------
    # 4. TYPE CLEANING
    # ------------------------------------------------------------------
    df["type"] = df["type"].astype(str).str.upper()

    # ------------------------------------------------------------------
    # 5. NUMERIC SAFETY (NO MORE pd.to_numeric ERRORS)
    # ------------------------------------------------------------------
    numeric_cols = [
        "step",
        "amount",
        "oldbalanceOrig",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    for col in numeric_cols:
        df[col] = _ensure_series(df, col)

    # ------------------------------------------------------------------
    # 6. FEATURE ENGINEERING (MATCHES TRAINING)
    # ------------------------------------------------------------------
    df["errorBalanceOrig"] = (
        df["oldbalanceOrig"] - df["amount"] - df["newbalanceOrig"]
    )

    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    df["has_balance_mismatch"] = (
        (df["errorBalanceOrig"].abs() > 1)
        | (df["errorBalanceDest"].abs() > 1)
    ).astype(int)

    # ------------------------------------------------------------------
    # 7. TEMPORAL FEATURES
    # ------------------------------------------------------------------
    df["hour"] = (df["step"] % 24).astype(int)
    df["dayofweek"] = ((df["step"] // 24) % 7).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # ------------------------------------------------------------------
    # 8. TRANSACTION TYPE ONE-HOTS
    # ------------------------------------------------------------------
    df["upi_type_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["upi_type_cash_out"] = (df["type"] == "CASH_OUT").astype(int)
    df["upi_type_payment"] = (df["type"] == "PAYMENT").astype(int)

    # ------------------------------------------------------------------
    # 9. BACKWARD COMPATIBILITY (MODEL EXPECTS THIS NAME)
    # ------------------------------------------------------------------
    df["oldbalanceOrg"] = df["oldbalanceOrig"]


        # ------------------------------------------------------------------
    # 10. FINAL FEATURE NAME ALIGNMENT (STAGE-2 COMPATIBILITY)
    # ------------------------------------------------------------------
    rename_stage2 = {
        "oldbalanceOrig": "oldbalanceorg",
        "newbalanceOrig": "newbalanceorig",
        "oldbalanceDest": "oldbalancedest",
        "newbalanceDest": "newbalancedest",
        "errorBalanceOrig": "errorbalanceorig",
        "errorBalanceDest": "errorbalancedest",

        # UPI transaction types (Stage-2 expects these exact names)
        "upi_type_payment": "upi_type_upi_payment",
        "upi_type_transfer": "upi_type_upi_transfer",
    }

    # Handle potential duplicates correctly by dropping AFTER renaming
    df.rename(columns=rename_stage2, inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep='last')].copy()

    # Ensure missing Stage-2 columns exist (safety)
    for col in rename_stage2.values():
        if col not in df.columns:
            df[col] = 0


    return df
