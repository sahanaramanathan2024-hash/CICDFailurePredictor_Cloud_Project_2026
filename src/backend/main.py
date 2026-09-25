import os
import json
import time
import uuid
import boto3
import joblib
import xgboost as xgb
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load environment configuration before AWS client instantiation
load_dotenv()

app = FastAPI(
    title="CI/CD Failure Predictor & DevOps Analytics API",
    description="Predicts CI/CD pipeline failure risk and logs results to AWS.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration read from .env — no hardcoded keys, no hardcoded resource names
REGION = os.getenv("AWS_REGION", "ap-south-1")
BUCKET = os.getenv("S3_BUCKET")
LOCAL_DIR = "artifacts_cache"
LOG_GROUP = "/cicd-predictor/predictions"
LOG_STREAM = "api-stream"
DYNAMO_TABLE = os.getenv("DYNAMODB_TABLE")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")
ALERT_THRESHOLD = 0.70

os.makedirs(LOCAL_DIR, exist_ok=True)

# boto3 reads credentials automatically from ~/.aws/credentials (set by `aws configure`)
# Never hardcode access keys here.
s3 = boto3.client("s3", region_name=REGION)
logs_client = boto3.client("logs", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)
table = dynamodb.Table(DYNAMO_TABLE)


def ensure_log_group():
    try:
        logs_client.create_log_group(logGroupName=LOG_GROUP)
    except logs_client.exceptions.ResourceAlreadyExistsException:
        pass
    try:
        logs_client.create_log_stream(logGroupName=LOG_GROUP, logStreamName=LOG_STREAM)
    except logs_client.exceptions.ResourceAlreadyExistsException:
        pass


def download_artifacts():
    print("--- Syncing with AWS S3: pulling model assets ---")
    for key in ["model/model.json", "model/encoders.pkl", "model/feature_columns.json"]:
        target_path = os.path.join(LOCAL_DIR, os.path.basename(key))
        s3.download_file(BUCKET, key, target_path)


@app.on_event("startup")
def initialize_and_warm_inference_engine():
    global model, encoders, feature_cols
    ensure_log_group()
    download_artifacts()

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(LOCAL_DIR, "model.json"))
    encoders = joblib.load(os.path.join(LOCAL_DIR, "encoders.pkl"))
    with open(os.path.join(LOCAL_DIR, "feature_columns.json"), "r") as f:
        feature_cols = json.load(f)

    print(f"✅ Startup complete. Model loaded with {len(feature_cols)} features from S3.")


class BuildFeatures(BaseModel):
    features: dict = Field(..., example={
        "repo": "CICDFailurePredictor_Cloud_Project_2026", "event": "push", "head_branch": "develop",
        "run_number": 14, "author_name": "Sahana Ramanathan", "actor_login": "sahanaramanathan2024",
        "triggering_actor_login": "sahanaramanathan2024", "msg_len": 48, "is_merge": 0, "num_parents": 1,
        "additions": 145, "deletions": 32, "total_churn": 177, "files_modified": 3, "run_attempt": 1,
        "time_since_last_commit": 7200.0, "hour_of_day": 19, "day_of_week": 3, "is_weekend": 0,
        "repo_prior_failure_rate": 0.0594, "author_prior_failure_rate": 0.0412
    })


@app.post("/predict")
def predict(payload: BuildFeatures):
    try:
        current_time = int(time.time())
        prediction_id = str(uuid.uuid4())

        input_data = payload.features
        input_df = pd.DataFrame([input_data])

        categorical_targets = ["repo", "event", "head_branch", "author_name", "actor_login", "triggering_actor_login"]
        for col in categorical_targets:
            if col in input_df.columns and col in encoders:
                le = encoders[col]
                val = str(input_df[col].iloc[0])
                input_df[col] = le.transform([val]) if val in le.classes_ else le.transform([le.classes_[0]])

        row_features = input_df[feature_cols]
        prob = float(model.predict_proba(row_features)[0][1])
        outcome = "fail" if prob >= 0.50 else "pass"

        result = {"failure_probability": prob, "prediction": outcome, "prediction_id": prediction_id}

        table.put_item(Item={
            "prediction_id": prediction_id,
            "timestamp": current_time,
            "features": json.dumps(payload.features),
            "failure_probability": str(prob),
            "prediction": outcome
        })

        logs_client.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=LOG_STREAM,
            logEvents=[{
                "timestamp": int(time.time() * 1000),
                "message": json.dumps({"input": payload.features, "output": result})
            }]
        )

        if prob > ALERT_THRESHOLD:
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject="CI/CD Pipeline Failure Risk Alert",
                Message=f"High failure risk detected: {prob:.2%}\n\n"
                        f"Prediction ID: {prediction_id}\n"
                        f"Repository: {payload.features.get('repo')}\n"
                        f"Author: {payload.features.get('author_name')}"
            )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def history(limit: int = 20):
    try:
        resp = table.scan(Limit=limit)
        items = sorted(resp.get("Items", []), key=lambda x: int(x["timestamp"]), reverse=True)
        return {"count": len(items), "predictions": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    try:
        resp = table.scan()
        items = resp.get("Items", [])
        total = len(items)
        failures = sum(1 for i in items if i["prediction"] == "fail")
        return {
            "total_predictions_logged": total,
            "predicted_failures_flagged": failures,
            "failure_rate_percentage": (failures / total) if total else 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "features_loaded": len(feature_cols)}