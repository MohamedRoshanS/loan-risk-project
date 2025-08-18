# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from .schemas import Borrower
import pandas as pd
import joblib
import shap

from src.features import engineer_features

app = FastAPI(title="AI-Powered Loan Risk API")

# Load trained model pipeline
MODEL_PATH = "api/model.joblib"
pipeline = joblib.load(MODEL_PATH)

# Initialize SHAP explainer (TreeExplainer works for XGBoost, CatBoost, LightGBM)
explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])

@app.get("/")
def root():
    return {"message": "Loan Risk API is running"}

@app.post("/predict/")
def predict_risk(borrower: Borrower):
    try:
        # Convert Pydantic model to DataFrame
        df = pd.DataFrame([borrower.dict()])

        # Engineer features exactly as in training
        df = engineer_features(df)

        # Predict probability of default (class 1)
        risk_prob = pipeline.predict_proba(df)[:, 1][0]

        # SHAP explanation
        preprocessor = pipeline.named_steps['preprocessor']
        X_processed = preprocessor.transform(df)

        # Extract feature names after preprocessing
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = list(num_features) + list(cat_features)

        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_processed)[0]

        # Prepare top-5 feature impacts
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "impact": shap_values
        }).assign(abs_impact=lambda x: x['impact'].abs())
        top_features = shap_df.sort_values('abs_impact', ascending=False).head(5)
        explanation = top_features[['feature', 'impact']].to_dict(orient="records")

        return {
            "risk_score": float(risk_prob),
            "reason": explanation
        }

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
