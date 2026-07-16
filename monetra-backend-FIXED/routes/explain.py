"""routes/explain.py — GET /explain  (RAG + OpenAI)"""
from fastapi import APIRouter, Depends, Query
from database import get_db
from models.schemas import ExplainResponse
from services.drift_service import run_full_drift_analysis
from services.risk_service import build_risk_summary
from services.explain_service import generate_explanation
from services.seed_service import load_reference_stats

router = APIRouter()

@router.get("/explain", response_model=ExplainResponse)
def explain(window: int = Query(default=200, ge=10, le=2000), conn=Depends(get_db)):
    from utils.model_loader import get_active_model_key
    active_key = get_active_model_key()

    cur = conn.execute(
        """SELECT loan_amount, annual_income, credit_score,
                  applicant_age, loan_tenure,
                  employment_type, prediction, confidence
           FROM prediction_logs WHERE model_key=? ORDER BY id DESC LIMIT ?""", (active_key, window)
    )
    recent_preds = [dict(r) for r in cur.fetchall()]

    reference_stats = load_reference_stats()
    drift_summary   = run_full_drift_analysis(recent_preds, reference_stats)
    risk_summary    = build_risk_summary(recent_preds, drift_summary)
    explanation     = generate_explanation(drift_summary, risk_summary, recent_preds)

    return ExplainResponse(
        summary         = explanation["summary"],
        severity        = explanation.get("severity", "LOW"),
        root_causes     = explanation["root_causes"],
        recommendations = explanation["recommendations"],
        risk_level      = explanation["risk_level"],
        urgency         = explanation.get("urgency", "Monitor"),
        generated_at    = explanation["generated_at"],
        context_used    = explanation.get("context_used", []),
        source          = explanation.get("source", "rule-based"),
    )
