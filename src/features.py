from __future__ import annotations
import numpy as np
import pandas as pd


# -----------------------------
# Numeric Feature Engineering
# -----------------------------
def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Income" in df.columns:
        df["LogIncome"] = np.log1p(df["Income"])
    if "LoanAmount" in df.columns:
        df["LogLoanAmount"] = np.log1p(df["LoanAmount"])
    return df


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ✅ Compute DTI ratio if not already available
    if "LoanAmount" in df.columns and "Income" in df.columns:
        df["DTIRatio"] = df["LoanAmount"] / df["Income"].replace(0, np.nan)
        df["DTIRatio"] = df["DTIRatio"].fillna(0)

    # Loan-to-Income
    if "LoanAmount" in df.columns and "Income" in df.columns:
        df["LTI"] = df["LoanAmount"] / df["Income"].replace(0, np.nan)
        df["LTI"] = df["LTI"].fillna(0)

    # Payment-to-Income
    if all(col in df.columns for col in ["LoanAmount", "InterestRate", "Income"]):
        df["PTI"] = (df["LoanAmount"] * (1 + df["InterestRate"] / 100)) / df["Income"].replace(0, np.nan)
        df["PTI"] = df["PTI"].fillna(0)

    return df


def bin_credit_score(df: pd.DataFrame) -> pd.DataFrame:
    if "CreditScore" not in df.columns:
        return df
    df = df.copy()
    bins = [300, 500, 650, 750, 850]
    labels = ["Poor", "Fair", "Good", "Excellent"]
    df["CreditScoreBand"] = pd.cut(
        df["CreditScore"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    return df


def bin_age(df: pd.DataFrame) -> pd.DataFrame:
    if "Age" not in df.columns:
        return df
    df = df.copy()
    bins = [18, 25, 35, 45, 55, 69]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-69"]
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    return df


# -----------------------------
# Interaction Features
# -----------------------------
def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LoanTerm_safe"] = df["LoanTerm"].replace(0, 1)

    df["EMI"] = df["LoanAmount"] / df["LoanTerm_safe"]
    df["Interest_Loan"] = df["InterestRate"] * df["LoanAmount"]

    df["Income_HasDependents"] = df["Income"] * df["HasDependents"].map(lambda x: 1 if x == "Yes" else 0)
    df["DTI_HasMortgage"] = df["DTIRatio"] * df["HasMortgage"].map(lambda x: 1 if x == "Yes" else 0)
    df["Interest_LTI"] = df["InterestRate"] * df["LTI"]
    df["MonthsEmp_CreditScore"] = df["MonthsEmployed"] * df["CreditScore"]

    df = df.drop(columns=["LoanTerm_safe"])
    return df


# -----------------------------
# Master function
# -----------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_log_transforms(df)
    df = add_ratios(df)          # ✅ ensures DTIRatio, LTI, PTI
    df = bin_credit_score(df)
    df = bin_age(df)
    df = add_interactions(df)

    # Drop LoanID if present
    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])

    return df
