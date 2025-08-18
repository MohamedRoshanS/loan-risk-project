# AI-Powered Loan Eligibility & Risk Scoring System

An end-to-end machine learning solution for predicting loan default risk, featuring automated risk scoring, interpretability, and a robust FastAPI backend for production deployment.

---

## 🚀 Overview

Traditional lending relies on fixed rules, but this system enables:
- **Real-time risk scoring** for loan applicants
- **Feature-level explanations** for transparency
- **Scalable API endpoints** ready for production
- **Comprehensive EDA & charts** to support informed decisions

---

## 📂 Project Structure

```

project-root/
├── api/
│   ├── main.py           \# FastAPI backend
│   ├── model.joblib      \# Trained ML model
│   └── schemas.py        \# Pydantic validation
├── charts/               \# Saved visualizations (.png, .jpg)
├── notebooks/
│   └── exploration.ipynb \# EDA \& feature engineering
├── data/
│   └── 6S_AI_TASK-Loan_default_Loan_de.xlsx  \# Raw dataset
├── requirements.txt      \# Python dependencies
└── README.md             \# Project documentation

```

---

## 📊 Dataset

- **Source:** Excel file: `6S_AI_TASK-Loan_default_Loan_de.xlsx`
- **Rows:** 255,246 **Columns:** 18
- **Target:** `Default` (0 = safe, 1 = default)
- **Imbalance:** 1 = 11.6%; 0 = 88.4%
- **Features:** Age, Income, LoanAmount, CreditScore, InterestRate, MonthsEmployed, DTIRatio, EmploymentType, MaritalStatus, etc.
- Full EDA: See `notebooks/exploration.ipynb`

---

## ⚙️ Features & ML Pipeline

- **Preprocessing:** Missing values, duplicates, outliers
- **Feature Engineering:**
  - Log transforms: Income, LoanAmount  
  - Ratios: Loan-to-Income (LTI), Payment-to-Income (PTI)
  - Binning: CreditScoreBand, AgeBand, InterestRateBand
  - Interaction: DTI × HasMortgage, Income × HasDependents
- **Models:** Gradient Boosted (CatBoost, XGBoost), hyperparameter tuning
- **Metrics:** ROC-AUC, PR-AUC, F1, Precision, Recall, Calibration

---

## 🟢 FastAPI Backend

**Endpoints:**
- `/predict/` Predict default probability and provide feature impact explanation
- `/health`  API status check

**Features:**
- Pydantic input validation (types, ranges)
- Informative error messages

---

## 🛠️ Installation & Setup

```

git clone <repo_url>
cd <repo_folder>

python -m venv venv
source venv/bin/activate    \# On Windows: venv\Scripts\activate

pip install -r requirements.txt

# (Optional) Train model if needed:

python notebooks/train_model.py

# Run backend:

uvicorn api.main:app --reload

```

---

## 📑 API Usage

### 1️⃣ Health Check

**GET** `/health`

**Response:**
```

{"status": "ok"}

```

---

### 2️⃣ Predict Risk Score

**POST** `/predict/`

**Request Example:**
```

{
"Age": 35,
"Income": 90000,
"LoanAmount": 15000,
"CreditScore": 720,
"InterestRate": 6.5,
"MonthsEmployed": 120,
"DTIRatio": 0.25,
"EmploymentType": "Salaried",
"MaritalStatus": "Single"
}

```

**Response Example:**
```

{
"risk_score": 0.23,
"default_prediction": 0,
"reason": [
{"feature": "Income", "impact": -0.12},
{"feature": "CreditScore", "impact": -0.09},
{"feature": "LoanAmount", "impact": 0.06}
]
}

```
- `reason` explains top factors: negative values reduce risk, positive increase.

---

## 📈 Visualizations

- Numeric & categorical distributions
- Correlation heatmaps
- Class imbalance plots
- SHAP feature importances
- Segmentation by risk strata
- Charts saved in `/charts/` Documented in EDA notebook

---

## 🏆 Evaluation Metrics

| Metric             | Value |
|--------------------|-------|
| Accuracy           | 0.80  |
| F1-score           | 0.37  |
| ROC-AUC            | 0.75  |
| Precision (Def=1)  | 0.29  |
| Recall (Def=1)     | 0.50  |

---

## 📌 Key Insights

- Older age, higher income, smaller loans, and higher credit scores reduce risk
- DTIRatio, InterestRate, MonthsEmployed are highly predictive
- SHAP explanations deliver transparent, auditable risk scoring

---

## 🔮 Future Improvements

- Integrate real-time credit bureau data
- Add authentication & logging for secure API access
- Deploy model ensembles for higher accuracy
- Continuously retrain with new loan data

---

## 📝 References

- FastAPI Documentation
- CatBoost Python Guide
- SHAP Explainability

---

**For details on data, code, or deployment, see `/notebooks/exploration.ipynb` and API code in `/api/`.**
```