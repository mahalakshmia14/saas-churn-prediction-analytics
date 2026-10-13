import pandas as pd
import numpy as np
import json
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, confusion_matrix,
                              roc_curve, precision_recall_curve, classification_report)

RANDOM_SEED = 42
OUT = "/home/claude/project/outputs"
MODELS = "/home/claude/project/models"
os.makedirs(MODELS, exist_ok=True)
os.makedirs(f"{OUT}/reports", exist_ok=True)

# ----------------------------------------------------------------------
# Load master dataset & define leakage-safe feature set
# ----------------------------------------------------------------------
m = pd.read_parquet(f"{OUT}/customer_master.parquet")

# Columns intentionally EXCLUDED from modeling (identifiers / non-features):
ID_COLS = ["account_id", "reference_date"]
TARGET = "churn"

CATEGORICAL = [
    "industry", "country", "referral_source", "initial_plan_tier",
    "current_plan_tier", "current_billing_frequency",
]
BOOLEAN = [
    "account_is_trial", "annual_billing_ever", "auto_renew_flag", "currently_trialing",
]
NUMERIC = [
    "initial_seats", "tenure_days", "subscription_count", "total_upgrades", "total_downgrades",
    "max_seats_seen", "current_mrr", "current_arr", "current_seats", "upgrade_rate", "downgrade_rate",
    "total_usage_count", "total_usage_duration_secs", "total_error_count", "distinct_features_used",
    "beta_feature_events", "usage_event_count", "feature_adoption_rate", "error_rate",
    "beta_feature_rate", "recent_30d_usage", "prior_30d_usage", "usage_decline_rate",
    "usage_per_tenure_day", "total_tickets", "avg_resolution_time_hours", "avg_first_response_minutes",
    "avg_satisfaction_score", "pct_missing_satisfaction", "escalation_count", "high_urgent_tickets",
    "has_satisfaction_data", "tickets_per_30_tenure_days", "escalation_rate",
]

FEATURES = CATEGORICAL + BOOLEAN + NUMERIC
assert set(FEATURES) <= set(m.columns), set(FEATURES) - set(m.columns)

for c in BOOLEAN:
    m[c] = m[c].astype(int)

X = m[FEATURES].copy()
y = m[TARGET].copy()

print("Feature matrix:", X.shape, "| Class balance:", y.value_counts(normalize=True).to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)
print("Train:", X_train.shape, "Test:", X_test.shape)

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC + BOOLEAN),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                             class_weight="balanced", random_state=RANDOM_SEED),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_SEED,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    ),
}

results = {}
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
    cm = confusion_matrix(y_test, pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)

    results[name] = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    fitted_pipelines[name] = pipe
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

# ----------------------------------------------------------------------
# Select best model objectively (ROC-AUC primary, recall as tiebreaker per project brief)
# ----------------------------------------------------------------------
best_name = max(results, key=lambda n: (results[n]["metrics"]["roc_auc"]))
print(f"\nBest model by ROC-AUC: {best_name}")

best_pipeline = fitted_pipelines[best_name]
joblib.dump(best_pipeline, f"{MODELS}/trained_model.pkl")
joblib.dump({"features": FEATURES, "categorical": CATEGORICAL, "boolean": BOOLEAN,
             "numeric": NUMERIC, "best_model": best_name}, f"{MODELS}/model_metadata.pkl")

for name, pipe in fitted_pipelines.items():
    fname = name.lower().replace(" ", "_")
    joblib.dump(pipe, f"{MODELS}/{fname}.pkl")

with open(f"{OUT}/reports/model_results.json", "w") as f:
    json.dump({"best_model": best_name, "results": results,
                "test_size": len(X_test), "train_size": len(X_train)}, f, indent=2)

# Save test set + predictions from best model for dashboard use
X_test_out = X_test.copy()
X_test_out["account_id"] = m.loc[X_test.index, "account_id"].values
X_test_out["churn_actual"] = y_test.values
X_test_out["churn_probability"] = best_pipeline.predict_proba(X_test)[:, 1]
X_test_out.to_parquet(f"{OUT}/test_predictions.parquet")

# Score ALL customers with the best model (used by the live dashboard risk table)
all_proba = best_pipeline.predict_proba(X)[:, 1]
scored = m[["account_id", "industry", "current_plan_tier", "current_mrr", "tenure_days",
            "total_tickets", "feature_adoption_rate", "usage_decline_rate", "churn"]].copy()
scored["churn_probability"] = all_proba


def risk_level(p):
    if p < 0.30:
        return "Low"
    elif p < 0.60:
        return "Medium"
    elif p < 0.80:
        return "High"
    else:
        return "Critical"


scored["risk_level"] = scored["churn_probability"].apply(risk_level)
scored["revenue_at_risk"] = scored["churn_probability"] * scored["current_mrr"]
scored.to_parquet(f"{OUT}/scored_customers.parquet")
scored.to_csv(f"{OUT}/reports/scored_customers.csv", index=False)

print("\nSaved: model pkl files, model_results.json, test_predictions.parquet, scored_customers.parquet")
