"""
Phase 2 - Feature Engineering + ML Model
CI/CD Pipeline Failure Prediction (XGBoost + SHAP)

Input:  dataset/processed/clean_builds.csv (40,000 rows, GADFPD schema)
Output: src/ml_model/model_artifact/{model.json, encoders.pkl, feature_columns.json}
        results/accuracy.xlsx, results/graphs/*.png

Run from repo root: python src/ml_model/train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
)
from xgboost import XGBClassifier
import shap

DATA_PATH = "dataset/processed/clean_builds.csv"
ARTIFACT_DIR = "src/ml_model/model_artifact"
GRAPH_DIR = "results/graphs"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Failure rate: {df['build_failed'].mean():.4f}")

# ---------------------------------------------------------------------------
# 2. DROP LEAKAGE / ID / EMPTY COLUMNS
# ---------------------------------------------------------------------------
# These are only known AFTER the run completes, or are pure identifiers,
# or (repo_name, commit_date, author, timestamp) are entirely NaN in this export.
LEAKAGE_OR_USELESS = [
    "run_id", "commit_sha", "conclusion", "status", "duration",
    "updated_at", "node_id", "head_sha", "repo_name",
    "commit_date", "author", "timestamp", "workflow_id",
    "author_email", "committer_name", "committer_email",
    "message", "authored_date", "committed_date", "timestamp_raw",
]
df = df.drop(columns=[c for c in LEAKAGE_OR_USELESS if c in df.columns])

# ---------------------------------------------------------------------------
# 3. TIME-BASED FEATURES (pre-run info: when the build was triggered)
# ---------------------------------------------------------------------------
df["run_started_at"] = pd.to_datetime(df["run_started_at"], utc=True, errors="coerce")
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

df["hour_of_day"] = df["run_started_at"].dt.hour
df["day_of_week"] = df["run_started_at"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df = df.drop(columns=["run_started_at", "created_at"])

# ---------------------------------------------------------------------------
# 4. CAUSAL "PRIOR FAILURE RATE PER REPO" (expanding mean, shifted by 1
#    so a build never sees its own or future outcomes -> no leakage)
# ---------------------------------------------------------------------------
df = df.sort_values("run_number").reset_index(drop=True)
df["repo_prior_failure_rate"] = (
    df.groupby("repo")["build_failed"]
      .transform(lambda s: s.shift().expanding().mean())
)
# first build of each repo has no history -> fill with global mean
df["repo_prior_failure_rate"] = df["repo_prior_failure_rate"].fillna(
    df["build_failed"].mean()
)

# same idea for the author (their personal historical failure rate)
df["author_prior_failure_rate"] = (
    df.groupby("author_name")["build_failed"]
      .transform(lambda s: s.shift().expanding().mean())
)
df["author_prior_failure_rate"] = df["author_prior_failure_rate"].fillna(
    df["build_failed"].mean()
)

# ---------------------------------------------------------------------------
# 5. ENCODE CATEGORICALS
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = ["repo", "event", "head_branch", "author_name",
                     "actor_login", "triggering_actor_login"]

encoders = {}
for col in CATEGORICAL_COLS:
    df[col] = df[col].astype(str).fillna("unknown")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df["is_merge"] = df["is_merge"].astype(int)

# ---------------------------------------------------------------------------
# 6. FINAL FEATURE SET
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "repo", "event", "head_branch", "run_number",
    "author_name", "actor_login", "triggering_actor_login",
    "msg_len", "is_merge", "num_parents", "additions", "deletions",
    "total_churn", "files_modified", "run_attempt", "time_since_last_commit",
    "hour_of_day", "day_of_week", "is_weekend",
    "repo_prior_failure_rate", "author_prior_failure_rate",
]
TARGET_COL = "build_failed"

df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median(numeric_only=True))

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\nFinal feature matrix: {X.shape}")
print(f"Features: {FEATURE_COLS}")

# ---------------------------------------------------------------------------
# 7. TRAIN/TEST SPLIT (stratified so both sets keep the ~5.9% failure ratio)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain failure rate: {y_train.mean():.4f}")
print(f"Test failure rate:  {y_test.mean():.4f}")

# ---------------------------------------------------------------------------
# 8. TRAIN XGBOOST (scale_pos_weight handles the class imbalance)
# ---------------------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    early_stopping_rounds=20,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ---------------------------------------------------------------------------
# 9. EVALUATE (report metrics on the FAILURE class specifically -- with
#    5.9% imbalance, overall accuracy is misleading)
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

report_dict = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, y_proba)
f1_fail = f1_score(y_test, y_pred, pos_label=1)
recall_fail = recall_score(y_test, y_pred, pos_label=1)
precision_fail = precision_score(y_test, y_pred, pos_label=1)

print("\n=== Evaluation (failure class = 1) ===")
print(f"Precision: {precision_fail:.4f}")
print(f"Recall:    {recall_fail:.4f}")
print(f"F1:        {f1_fail:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print(classification_report(y_test, y_pred))

# Save metrics to Excel
metrics_df = pd.DataFrame({
    "metric": ["precision_failure", "recall_failure", "f1_failure", "auc_roc",
               "accuracy_overall"],
    "value": [precision_fail, recall_fail, f1_fail, auc, report_dict["accuracy"]],
})
metrics_df.to_excel("results/accuracy.xlsx", index=False)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Success", "Failure"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - CI/CD Failure Prediction")
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# ROC curve plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - CI/CD Failure Prediction")
plt.legend()
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 10. SHAP EXPLAINABILITY
# ---------------------------------------------------------------------------
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Local explanation for one predicted-failure sample
failure_idx = np.where(y_pred == 1)[0]
if len(failure_idx) > 0:
    sample_i = failure_idx[0]
    plt.figure()
    shap.force_plot(
        explainer.expected_value, shap_values[sample_i], X_test.iloc[sample_i],
        matplotlib=True, show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/shap_force_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
else:
    print("No predicted failures in test set to plot a force plot for.")

# ---------------------------------------------------------------------------
# 11. SAVE ARTIFACTS (backend / Phase 3 consumes these three files)
# ---------------------------------------------------------------------------
model.save_model(f"{ARTIFACT_DIR}/model.json")
joblib.dump(encoders, f"{ARTIFACT_DIR}/encoders.pkl")
with open(f"{ARTIFACT_DIR}/feature_columns.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

print(f"\nArtifacts saved to {ARTIFACT_DIR}/")
print("  - model.json")
print("  - encoders.pkl")
print("  - feature_columns.json")
print(f"\nMetrics saved to results/accuracy.xlsx")
print(f"Plots saved to {GRAPH_DIR}/")
print("\nPhase 2 complete.")