from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier


def build_pipeline():
    """
    Returns a preprocessing + CatBoost pipeline for the loan dataset.
    """

    num_features = [
        "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
        "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
        "LogIncome", "LogLoanAmount", "LTI", "PTI",
        "EMI", "Interest_Loan", "Income_HasDependents",
        "DTI_HasMortgage", "Interest_LTI", "MonthsEmp_CreditScore"
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

    # ✅ CatBoost only
    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=8,
        eval_metric="F1",
        verbose=0,
        random_state=42,
        class_weights=[1, 10],
        early_stopping_rounds=50
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline
