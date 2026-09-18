"""S3 helpers — Day 1 task 2/3 (Pranav).

Bucket: cicd-failure-predictor-data (single shared bucket for the team).
Usage:
    python -m src.aws.s3_utils upload dataset/processed/train.csv train.csv
    python -m src.aws.s3_utils download train.csv /tmp/train.csv
"""
from __future__ import annotations

import os
import sys

BUCKET = os.getenv("S3_BUCKET", "cicd-failure-predictor-data")
REGION = os.getenv("AWS_REGION", "ap-south-1")


def _client():
    import boto3

    return boto3.client("s3", region_name=REGION)


def upload_file(local_path: str, key: str, bucket: str = BUCKET) -> str:
    s3 = _client()
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def read_key(key: str, bucket: str = BUCKET, limit: int = 500) -> bytes:
    s3 = _client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read(limit)


def upload_build_payload(payload: dict, key: str, bucket: str = BUCKET) -> str:
    """Upload a single build feature JSON (used by CI webhook before Lambda fires)."""
    import json

    s3 = _client()
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode(), ContentType="application/json")
    return f"s3://{bucket}/{key}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: s3_utils.py upload <local> <key> | download <key> <local>")
        sys.exit(1)
    if sys.argv[1] == "upload":
        print(upload_file(sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == "download":
        data = read_key(sys.argv[2], limit=10_000_000)
        open(sys.argv[3], "wb").write(data)
        print(f"wrote {len(data)} bytes to {sys.argv[3]}")
    else:
        # Day-1 verification snippet from the guide
        import boto3

        s3 = boto3.client("s3")
        s3.upload_file("dataset/processed/train.csv", BUCKET, "train.csv")
        obj = s3.get_object(Bucket=BUCKET, Key="train.csv")
        print(obj["Body"].read()[:200])
