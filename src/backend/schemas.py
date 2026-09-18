"""Pydantic schemas — must match src/ml_model/predict.py FEATURE_COLUMNS exactly."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class BuildFeatures(BaseModel):
    total_churn: float = Field(..., ge=0, description="src+test churn lines")
    files_touched: float = Field(..., ge=0)
    is_pr: int = Field(..., ge=0, le=1)
    project_rolling_failure_rate: float = Field(..., ge=0, le=1)
    num_commits_in_push: float = Field(default=1, ge=0)
    test_churn_ratio: float = Field(default=0, ge=0, le=1)
    prev_build_failed: int = Field(default=0, ge=0, le=1)
    team_past_failure_rate: float = Field(default=0, ge=0, le=1)
    stage_test_duration_avg: float = Field(default=0, ge=0)
    num_failed_tests_prev: float = Field(default=0, ge=0)
    lines_added: float = Field(default=0, ge=0)
    build_hour: float = Field(default=12, ge=0, le=23)

    # Fleet-level analytics dimensions (never secondary — first-class fields)
    repo: str = Field(default="unknown", description="repo slug, e.g. apache/commons-lang")
    team: str = Field(default="unknown")
    pipeline_stage: str = Field(default="build", description="build|test|deploy")
    build_id: Optional[str] = None


class PredictResponse(BaseModel):
    prediction: Literal["passed", "failed"]
    probability: float
    model_version: str
    build_id: Optional[str] = None
    mock: bool = False


class FeedbackRequest(BaseModel):
    build_id: str
    actual: Literal["passed", "failed"]
