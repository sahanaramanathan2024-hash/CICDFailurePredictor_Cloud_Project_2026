"""Structured JSON logging — CloudWatch-ready.

Locally: human-readable JSON lines on stdout.
On AWS: CloudWatch picks up stdout from Lambda/EC2/ECS; log group configured
in src/aws/cloudwatch_config.json. Fields follow a fixed schema so QuickSight
and CloudWatch Insights queries work unchanged.
"""
from __future__ import annotations

import json
import logging
import os
import sys

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        for key in ("request_id", "repo", "team", "stage", "probability", "prediction"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)[:2000]
        return json.dumps(payload)


def get_logger(name: str = "cicd-predictor") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False
    return logger


log = get_logger()
