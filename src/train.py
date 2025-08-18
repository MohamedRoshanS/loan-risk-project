import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTENC
import joblib
from features import engineer_features
from pipeline import build_pipeline  # Import CatBoost pipeline

# -------------------------------
# Load dataset
# -------------------------------
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data/raw/dataset.xlsx"
PROC_DATA_PATH = Path(__file__).resolve().parents[1] / "data/processed/loan_features.csv"

# Load raw dataset
df = pd.read_excel(RAW_DATA_PATH)
print(f"[INFO] Raw dataset loaded: {RAW_DATA_PATH} ({df.shape} rows × {df.shape} columns)")

# -------------------------------
# Apply feature engineering
# -------------------------------
df = engineer_features(df)

# Separate features & target
X = df.drop(columns=["Default"])
y = df["Default"]

# Save processed dataset for inspection
PROC_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(PROC_DATA_PATH, index=False)
print(f"[INFO] Processed features saved at: {PROC_DATA_PATH}")

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
cat_cols = [
    "Education", "EmploymentType", "MaritalStatus", "LoanPurpose",
    "CreditScoreBand", "AgeBand", "HasMortgage", "HasDependents", "HasCoSigner"
]
cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
smote = SMOTENC(categorical_features=cat_indices, random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"[INFO] After SMOTENC, Train shape: {X_train_res.shape}, Class distribution: {y_train_res.value_counts().to_dict()}")

# -------------------------------
# Build & Train pipeline (CatBoost only)
# -------------------------------
pipeline = build_pipeline()
pipeline.fit(X_train_res, y_train_res)

# -------------------------------
# Threshold optimization for class 1
# -------------------------------
y_proba = pipeline.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-6)
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
