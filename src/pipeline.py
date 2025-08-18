# src/pipeline.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def build_pipeline(model_type="XGB"):
    """
    Returns a preprocessing + model pipeline for the loan dataset.
    model_type: "XGB", "LGBM", or "CatBoost"
    """
    num_features = [
        "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
        "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
        "LogIncome", "LogLoanAmount", "LTI", "PTI",
        "EMI", "Interest_Loan", "Income_HasDependents"
    ]

    cat_features = [
        "Education", "EmploymentType", "MaritalStatus", "LoanPurpose",
        "CreditScoreBand", "AgeBand", "HasMortgage", "HasDependents", "HasCoSigner"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )

    # Select model
    if model_type == "XGB":
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=5
        )
    elif model_type == "LGBM":
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight="balanced"
        )
    elif model_type == "CatBoost":
        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            eval_metric="F1",
            verbose=0,
            random_state=42,
            class_weights=[1,5]
        )
    else:
        raise ValueError(f"Unknown model_type={model_type}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline
