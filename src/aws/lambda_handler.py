import xgboost as xgb
import joblib
import json
import boto3
import os

LOCAL_DIR = "/tmp/artifacts"
os.makedirs(LOCAL_DIR, exist_ok=True)
s3 = boto3.client("s3")
BUCKET = "cicd-failure-predictor-sahana"

def load_artifacts():
    for key in ["model_artifact/model.json", "model_artifact/encoders.pkl", "model_artifact/feature_columns.json"]:
        local = os.path.join(LOCAL_DIR, os.path.basename(key))
        if not os.path.exists(local):
            s3.download_file(BUCKET, key, local)
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(LOCAL_DIR, "model.json"))
    feature_cols = json.load(open(os.path.join(LOCAL_DIR, "feature_columns.json")))
    return model, feature_cols

def lambda_handler(event, context):
    """
    AWS Lambda Inference Entry Point designed for serverless execution.
    Pending infrastructure container deployment due to XGBoost package size constraints (>250MB unzip limits).
    """
    model, feature_cols = load_artifacts()
    features = json.loads(event.get("body", "{}")).get("features", {})
    row = [features.get(c, 0) for c in feature_cols]
    prob = float(model.predict_proba([row])[0][1])
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"failure_probability": prob, "prediction": "fail" if prob > 0.5 else "pass"})
    }
