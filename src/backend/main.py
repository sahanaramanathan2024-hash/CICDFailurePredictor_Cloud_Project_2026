"""FastAPI backend — Pranav (feature/student2).

Day 1: mocked /predict (works with no model file).
Day 2: real model via src.ml_model.predict.predict (auto-used when model file exists).
Day 3: Bearer auth + Lambda trigger support.

Run: uvicorn src.backend.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.backend.auth import API_KEY, check_bearer
from src.backend.database import log_prediction, query_analytics, set_actual
from src.backend.logging_config import log
from src.backend.schemas import BuildFeatures, FeedbackRequest, PredictResponse
from src.ml_model.predict import predict as ml_predict

app = FastAPI(
    title="CICD Failure Predictor",
    version="0.3.0",
    description="Predicts CI/CD pipeline failures + fleet-level DevOps analytics by repo/team/stage.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _auth(x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None)):
    # Accept either X-API-Key or Authorization: Bearer <key>. Skip check if default dev key in use locally
    if os.getenv("REQUIRE_AUTH", "0") != "1":
        return True
    if x_api_key is not None and x_api_key == API_KEY:
        return True
    if check_bearer(authorization):
        return True
    raise HTTPException(status_code=401, detail="Unauthorized: provide X-API-Key or Bearer token")


@app.get("/health")
def health():
    return {"status": "ok", "service": "cicd-failure-predictor"}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(_auth)])
def predict(features: BuildFeatures):
    log.info("request received", extra={"repo": features.repo, "team": features.team, "stage": features.pipeline_stage})
    try:
        result = ml_predict(features.model_dump())
    except Exception as e:
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"model error: {e}")

    build_id = features.build_id or f"b-{uuid.uuid4().hex[:8]}"
    try:
        log_prediction(
            build_id=build_id,
            repo=features.repo,
            team=features.team,
            stage=features.pipeline_stage,
            features=features.model_dump(),
            prediction=result["prediction"],
            probability=result["probability"],
            model_version=result.get("model_version", ""),
        )
    except Exception as e:
        log.warning(f"db log failed: {e}")

    log.info(
        "prediction made",
        extra={
            "repo": features.repo,
            "prediction": result["prediction"],
            "probability": result["probability"],
        },
    )
    return PredictResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        model_version=result.get("model_version", ""),
        build_id=build_id,
        mock=result.get("mock", False),
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Susendar / CI system posts ground truth later — powers accuracy.xlsx + retraining."""
    n = set_actual(req.build_id, req.actual)
    if n == 0:
        raise HTTPException(status_code=404, detail="build_id not found")
    log.info(f"feedback recorded for {req.build_id}: {req.actual}")
    return {"updated": n, "build_id": req.build_id}


@app.get("/analytics/{group_by}")
def analytics(group_by: str):
    """Fleet-level DevOps analytics. group_by: repo|team|pipeline_stage.

    This is a first-class API, not a secondary add-on — QuickSight and the
    frontend dashboard consume this endpoint.
    """
    if group_by not in ("repo", "team", "pipeline_stage"):
        raise HTTPException(status_code=400, detail="group_by must be repo|team|pipeline_stage")
    return {"group_by": group_by, "rows": query_analytics(group_by)}


# Back-compat alias used by early frontend drafts
@app.get("/stats")
def stats():
    return {"by_repo": query_analytics("repo"), "by_team": query_analytics("team")}
