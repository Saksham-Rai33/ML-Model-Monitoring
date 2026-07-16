"""
routes/predict.py — POST /predict
Supports all 3 models: lasso, rf, xgb (default).
Pass ?model=lasso | rf | xgb in the query string.
"""
import numpy as np
from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException

from database import get_db
from models.schemas import LoanApplicationRequest, PredictionResponse
from utils.model_loader import get_model_info
from services.risk_service import compute_risk_level, infer_feature_contributions

router = APIRouter()

def _build_features_lasso(app: LoanApplicationRequest) -> np.ndarray:
    """15 features — Lasso notebook exact order."""
    edu = 0 if str(app.education).lower() in ("graduate", "0") else 1
    self_emp = 1 if str(app.self_employed).lower() in ("yes", "1") else 0
    total_assets = (app.residential_assets_value + app.commercial_assets_value +
                    app.luxury_assets_value + app.bank_asset_value)
    lti   = app.loan_amount / (app.income_annum + 1)
    atl   = total_assets / (app.loan_amount + 1)
    cibil_band = 0 if app.cibil_score<=549 else 1 if app.cibil_score<=649 else 2 if app.cibil_score<=749 else 3
    return np.array([[
        app.no_of_dependents, edu, self_emp,
        app.income_annum, app.loan_amount, app.loan_term, app.cibil_score,
        app.residential_assets_value, app.commercial_assets_value,
        app.luxury_assets_value, app.bank_asset_value,
        lti, total_assets, atl, cibil_band
    ]])

def _build_features_rf(app: LoanApplicationRequest) -> np.ndarray:
    """17 features — Random Forest notebook (adds income_per_dependent, emi_estimate)."""
    edu = 0 if str(app.education).lower() in ("graduate", "0") else 1
    self_emp = 1 if str(app.self_employed).lower() in ("yes", "1") else 0
    total_assets = (app.residential_assets_value + app.commercial_assets_value +
                    app.luxury_assets_value + app.bank_asset_value)
    lti  = app.loan_amount / (app.income_annum + 1)
    atl  = total_assets / (app.loan_amount + 1)
    cibil_band = 0 if app.cibil_score<=549 else 1 if app.cibil_score<=649 else 2 if app.cibil_score<=749 else 3
    ipd  = app.income_annum / (app.no_of_dependents + 1)
    emi  = app.loan_amount / (app.loan_term * 12 + 1)
    return np.array([[
        app.no_of_dependents, edu, self_emp,
        app.income_annum, app.loan_amount, app.loan_term, app.cibil_score,
        app.residential_assets_value, app.commercial_assets_value,
        app.luxury_assets_value, app.bank_asset_value,
        lti, total_assets, atl, cibil_band, ipd, emi
    ]])

def _build_features_xgb(app: LoanApplicationRequest) -> np.ndarray:
    """XGBoost features — all base + engineered, matching train_models.py order."""
    edu = 0 if str(app.education).lower() in ("graduate", "0") else 1
    self_emp = 1 if str(app.self_employed).lower() in ("yes", "1") else 0
    total_assets = (app.residential_assets_value + app.commercial_assets_value +
                    app.luxury_assets_value + app.bank_asset_value)
    lti  = app.loan_amount / (app.income_annum + 1)
    atl  = total_assets / (app.loan_amount + 1)
    ipd  = app.income_annum / (app.no_of_dependents + 1)
    emi  = app.loan_amount / (app.loan_term * 12 + 1)
    cibil_band = 0 if app.cibil_score<=549 else 1 if app.cibil_score<=649 else 2 if app.cibil_score<=749 else 3
    return np.array([[
        app.no_of_dependents, edu, self_emp,
        app.income_annum, app.loan_amount, app.loan_term, app.cibil_score,
        app.residential_assets_value, app.commercial_assets_value,
        app.luxury_assets_value, app.bank_asset_value,
        total_assets, lti, atl, ipd, emi, cibil_band
    ]])

FEATURE_BUILDERS = {
    "lasso": _build_features_lasso,
    "rf":    _build_features_rf,
    "xgb":  _build_features_xgb,
}

@router.post("/predict", response_model=PredictionResponse)
def predict(
    application: LoanApplicationRequest,
    model: str = Query(default="xgb", description="Model to use: lasso | rf | xgb"),
    conn = Depends(get_db)
):
    model_key = model.lower() if model.lower() in FEATURE_BUILDERS else "xgb"

    try:
        info = get_model_info(model_key)
        builder = FEATURE_BUILDERS[model_key]
        features = builder(application)

        clf = info["model"]

        if model_key == "xgb" and "scaler" in info:
            # XGBoost: separate scaler + model
            features = info["scaler"].transform(features)
            proba = clf.predict_proba(features)[0]
        elif hasattr(clf, 'predict_proba'):
            # Lasso/RF: sklearn Pipeline (scaler + classifier in one)
            proba = clf.predict_proba(features)[0]
        else:
            raise ValueError(f"Model {model_key} doesn't support predict_proba")

        prob_approve = float(proba[1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    prediction  = "APPROVED" if prob_approve > 0.5 else "REJECTED"
    confidence  = prob_approve if prediction == "APPROVED" else (1 - prob_approve)
    risk_score  = round(1 - confidence, 4)
    risk_level  = compute_risk_level((1 - risk_score) * 100)
    feature_weights = infer_feature_contributions(application.model_dump())

    cur = conn.execute(
        """INSERT INTO prediction_logs
           (loan_amount,annual_income,credit_score,applicant_age,loan_tenure,
            employment_type,prediction,confidence,risk_level,risk_score,model_key)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (application.loan_amount, application.income_annum, application.cibil_score,
         0, application.loan_term, application.self_employed,
         prediction, round(confidence, 4), risk_level, risk_score, model_key),
    )
    prediction_id = cur.lastrowid

    return PredictionResponse(
        prediction      = prediction,
        confidence      = round(confidence, 4),
        confidence_pct  = f"{confidence * 100:.1f}%",
        risk_level      = risk_level,
        risk_score      = risk_score,
        feature_weights = feature_weights,
        model_version   = info.get("version", "v1.0"),
        prediction_id   = prediction_id,
    )
