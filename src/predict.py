"""
Reusable scoring utility: load the trained pipeline and score a
customer-level dataframe (must contain the same engineered features
produced by feature_engineering.py).
"""
import pandas as pd
import joblib
import os

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")


def load_pipeline():
    pipeline = joblib.load(os.path.join(MODELS, "trained_model.pkl"))
    meta = joblib.load(os.path.join(MODELS, "model_metadata.pkl"))
    return pipeline, meta


def score_customers(df: pd.DataFrame) -> pd.DataFrame:
    """df: customer-level dataframe with the engineered feature columns.
    Returns df with churn_probability and risk_level columns added."""
    pipeline, meta = load_pipeline()
    X = df[meta["features"]].copy()
    for c in meta["boolean"]:
        X[c] = X[c].astype(int)

    proba = pipeline.predict_proba(X)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba

    def risk_level(p):
        if p < 0.30:
            return "Low"
        elif p < 0.60:
            return "Medium"
        elif p < 0.80:
            return "High"
        return "Critical"

    out["risk_level"] = out["churn_probability"].apply(risk_level)
    return out


if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
    master = pd.read_parquet(os.path.join(OUT, "customer_master.parquet"))
    scored = score_customers(master)
    print(scored[["account_id", "churn_probability", "risk_level"]].head(10))
