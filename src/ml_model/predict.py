"""Shared ML interface — owned by Sahana, consumed by Pranav's backend.

Contract (locked for Day 2 handoff):
- FEATURE_COLUMNS: exact order the model was trained on.
- predict(features: dict) -> dict with keys: prediction, probability, model_version
- Backend must call ml_predict(features_dict) and never reimplement feature logic.

Sahana: replace _load_model() + _mock_predict() with real XGBoost + SHAP logic.
Column mapping to TravisTorrent (Java only, gh_lang == 'Java', target tr_status):
"""
from __future__ import annotations

FEATURE_COLUMNS = [
    "total_churn",                    # git_diff_src_churn + git_diff_test_churn
    "files_touched",                  # gh_files_added + deleted + modified
    "is_pr",                          # gh_is_pr (0/1)
    "project_rolling_failure_rate",   # rolling mean of tr_status=failed per repo (last 20 builds)
    "num_commits_in_push",            # gh_num_commits_in_push
    "test_churn_ratio",               # test_churn / (total_churn + 1)
    "prev_build_failed",              # tr_status of previous build (0/1)
    "team_past_failure_rate",         # rolling failure rate per team
    "stage_test_duration_avg",        # avg tr_duration for test stage (seconds, log-scaled)
    "num_failed_tests_prev",          # tr_tests_failed on previous build
    "lines_added",                    # git_diff_src_added
    "build_hour",                     # hour of day build triggered (0-23)
]

MODEL_VERSION = "xgboost-v0.1-mock"
THRESHOLD = 0.5  # Sahana to tune on Day 2 (precision/recall tradeoff); backend reads this


def _load_model():
    """Try to load real model artefact. Returns None if not trained yet (Day 1)."""
    import os

    for candidate in [
        "src/ml_model/model.pkl",
        "src/ml_model/xgboost.json",
        "models/xgboost.json",
    ]:
        if os.path.exists(candidate):
            try:
                import pickle

                with open(candidate, "rb") as f:
                    return pickle.load(f)
            except Exception:
                continue
    return None


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _mock_predict(features: dict) -> dict:
    """Day 1 fallback heuristic so backend is never blocked. Remove once tuned."""
    score = 0.15
    score += min(float(features.get("total_churn", 0)) / 1000.0, 0.3)
    score += min(float(features.get("files_touched", 0)) / 20.0, 0.25)
    score += 0.2 * float(features.get("prev_build_failed", 0))
    score += 0.3 * float(features.get("project_rolling_failure_rate", 0))
    prob = max(0.02, min(0.97, score))
    return {
        "prediction": "failed" if prob >= THRESHOLD else "passed",
        "probability": round(prob, 4),
        "model_version": MODEL_VERSION,
        "mock": True,
    }


def predict(features: dict) -> dict:
    """Main entrypoint used by src/backend/main.py and src/aws/lambda_handler.py."""
    # Validate + order features
    ordered = {k: float(features.get(k, 0)) for k in FEATURE_COLUMNS}
    model = get_model()
    if model is None:
        return _mock_predict(ordered)
    try:
        import numpy as np

        X = np.array([[ordered[c] for c in FEATURE_COLUMNS]])
        proba = float(model.predict_proba(X)[0][1])
        return {
            "prediction": "failed" if proba >= THRESHOLD else "passed",
            "probability": round(proba, 4),
            "model_version": getattr(model, "version", "xgboost-real"),
            "mock": False,
        }
    except Exception as e:
        result = _mock_predict(ordered)
        result["fallback_error"] = str(e)
        return result
