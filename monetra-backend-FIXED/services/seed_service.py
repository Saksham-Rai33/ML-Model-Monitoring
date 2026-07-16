"""
services/seed_service.py — First-run initialisation
====================================================
Called once at app startup.
• Trains and saves the ML model if it doesn't exist.
• Populates the reference_stats table from training data statistics.
"""

import os
import logging
from database import get_conn

logger = logging.getLogger(__name__)

MODEL_PATH = "ml_models/loan_model.pkl"

REFERENCE_BASELINE = {
    "loan_amount":   {"mean": 420_000, "std": 180_000, "min_val": 50_000,  "max_val": 2_000_000, "p25": 280_000, "p50": 400_000, "p75": 550_000},
    "annual_income": {"mean": 850_000, "std": 350_000, "min_val": 100_000, "max_val": 5_000_000, "p25": 550_000, "p50": 780_000, "p75": 1_100_000},
    "credit_score":  {"mean": 680,     "std": 80,      "min_val": 300,     "max_val": 900,        "p25": 620,     "p50": 690,     "p75": 745},
    "applicant_age": {"mean": 35,      "std": 10,      "min_val": 18,      "max_val": 75,         "p25": 27,      "p50": 34,      "p75": 43},
    "loan_tenure":   {"mean": 36,      "std": 14,      "min_val": 6,       "max_val": 360,        "p25": 24,      "p50": 36,      "p75": 48},
}


def seed_reference_data():
    conn = get_conn()
    cur  = conn.execute("SELECT COUNT(*) FROM reference_stats")
    if cur.fetchone()[0] > 0:
        logger.info("Reference stats already seeded. Skipping.")
        _ensure_model_exists()
        return

    logger.info("Seeding reference statistics …")
    for feature, stats in REFERENCE_BASELINE.items():
        conn.execute(
            "INSERT OR IGNORE INTO reference_stats (feature,mean,std,min_val,max_val,p25,p50,p75) VALUES (?,?,?,?,?,?,?,?)",
            (feature, stats["mean"], stats["std"], stats["min_val"], stats["max_val"], stats["p25"], stats["p50"], stats["p75"]),
        )
    conn.commit()
    logger.info("Reference stats seeded successfully.")
    _ensure_model_exists()


def _ensure_model_exists():
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model already exists at {MODEL_PATH}.")
        return
    logger.info("No pre-trained model found. Training now …")
    try:
        from utils.model_trainer import train_and_save
        train_and_save(MODEL_PATH)
        logger.info("Model trained and saved.")
    except Exception as exc:
        logger.error(f"Model training failed: {exc}")


def load_reference_stats() -> dict:
    conn = get_conn()
    cur  = conn.execute("SELECT feature,mean,std,min_val,max_val,p25,p50,p75 FROM reference_stats")
    return {
        row["feature"]: {
            "mean": row["mean"], "std": row["std"],
            "min_val": row["min_val"], "max_val": row["max_val"],
            "p25": row["p25"], "p50": row["p50"], "p75": row["p75"],
        }
        for row in cur.fetchall()
    }
