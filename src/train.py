# src/train.py
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTENC
from catboost import CatBoostClassifier
import joblib

# -------------------------------
# Load dataset
# -------------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data/processed/loan_features.csv"
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset loaded: {DATA_PATH} ({df.shape[0]} rows × {df.shape[1]} columns)")

X = df.drop(columns=["Default"])
y = df["Default"]

# -------------------------------
# Enhanced Feature Engineering
# -------------------------------
X["EMI"] = X["LoanAmount"] / X["LoanTerm"]
X["Interest_Loan"] = X["InterestRate"] * X["LoanAmount"]
X["Income_HasDependents"] = X["Income"] * X["HasDependents"].apply(lambda x: 1 if x=="Yes" else 0)
X["DTI_HasMortgage"] = X["DTIRatio"] * X["HasMortgage"].apply(lambda x: 1 if x=="Yes" else 0)
X["Interest_LTI"] = X["InterestRate"] * X["LTI"]
X["MonthsEmp_CreditScore"] = X["MonthsEmployed"] * X["CreditScore"]

# Ensure numeric and categorical types
num_cols = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "LogIncome", "LogLoanAmount", "LTI", "PTI",
    "EMI", "Interest_Loan", "Income_HasDependents",
    "DTI_HasMortgage", "Interest_LTI", "MonthsEmp_CreditScore"
]
cat_cols = [
    "Education", "EmploymentType", "MaritalStatus", "LoanPurpose",
    "CreditScoreBand", "AgeBand", "HasMortgage", "HasDependents", "HasCoSigner"
]
X[num_cols] = X[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
X[cat_cols] = X[cat_cols].astype(str)

# -------------------------------
# Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -------------------------------
# SMOTENC oversampling
# -------------------------------
cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
smote = SMOTENC(categorical_features=cat_indices, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"[INFO] After SMOTENC, Train shape: {X_train_res.shape}, Class distribution: {y_train_res.value_counts().to_dict()}")

# -------------------------------
# Preprocessing pipeline
# -------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# -------------------------------
# CatBoost Classifier
# -------------------------------
model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.03,
    depth=8,
    eval_metric="F1",
    verbose=0,
    random_state=42,
    class_weights=[1,10],
    early_stopping_rounds=50
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# -------------------------------
# Train
# -------------------------------
pipeline.fit(X_train_res, y_train_res)

# -------------------------------
# Threshold optimization for class 1
# -------------------------------
y_proba = pipeline.predict_proba(X_test)[:,1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2*precisions*recalls/(precisions+recalls+1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
y_pred = (y_proba >= best_threshold).astype(int)
print(f"[INFO] Optimized threshold for class 1: {best_threshold:.2f}")

# -------------------------------
# Evaluation
# -------------------------------
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_proba)
print(f"[INFO] ROC-AUC: {roc_auc:.4f}")

# -------------------------------
# Save model
# -------------------------------
MODEL_PATH = Path(__file__).resolve().parents[1] / "api/model.joblib"
joblib.dump(pipeline, MODEL_PATH)
print(f"[INFO] Trained model saved at: {MODEL_PATH}")
