import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = "/home/claude/project/outputs"
MODELS = "/home/claude/project/models"
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

pipeline = joblib.load(f"{MODELS}/trained_model.pkl")
meta = joblib.load(f"{MODELS}/model_metadata.pkl")
m = pd.read_parquet(f"{OUT}/customer_master.parquet")

FEATURES = meta["features"]
X = m[FEATURES].copy()
for c in meta["boolean"]:
    X[c] = X[c].astype(int)

preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]
X_trans = preprocessor.transform(X)
if hasattr(X_trans, "toarray"):
    X_trans = X_trans.toarray()
feature_names = preprocessor.get_feature_names_out()
X_trans_df = pd.DataFrame(X_trans, columns=feature_names, index=X.index)

model_type = type(model).__name__
print("Best model:", meta["best_model"], "| sklearn class:", model_type)

if model_type == "LogisticRegression":
    explainer = shap.LinearExplainer(model, X_trans_df)
    shap_values = explainer.shap_values(X_trans_df)
elif model_type in ("RandomForestClassifier", "XGBClassifier"):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_trans_df)
    shap_values = sv[1] if isinstance(sv, list) else sv
else:
    background = shap.sample(X_trans_df, 50, random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_trans_df, nsamples=100)[1]

shap_values = np.array(shap_values)

# Global summary plot
plt.figure()
shap.summary_plot(shap_values, X_trans_df, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f"{FIG}/10_shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()

# Mean |SHAP| bar chart -> business-friendly "top drivers of churn risk"
mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)
mean_abs.head(15).to_csv(f"{OUT}/reports/shap_top_features.csv", header=["mean_abs_shap"])

plt.figure(figsize=(6, 6))
mean_abs.head(15).iloc[::-1].plot.barh(color="#C44E52")
plt.xlabel("Mean |SHAP value| (avg. impact on churn probability)")
plt.title("Top Churn Risk Drivers (SHAP)")
plt.tight_layout()
plt.savefig(f"{FIG}/11_shap_top_drivers.png", dpi=120)
plt.close()

# Per-customer explanations for the top-20 highest risk customers -> saved for dashboard
scored = pd.read_parquet(f"{OUT}/scored_customers.parquet")
top_risk = scored.sort_values("churn_probability", ascending=False).head(50)["account_id"]

explanations = {}
for acct in top_risk:
    idx = m.index[m["account_id"] == acct][0]
    pos = X.index.get_loc(idx)
    row_shap = pd.Series(shap_values[pos], index=feature_names)
    top_factors = row_shap.reindex(row_shap.abs().sort_values(ascending=False).index).head(5)
    explanations[acct] = [{"feature": f, "shap_value": float(v)} for f, v in top_factors.items()]

import json
with open(f"{OUT}/reports/shap_explanations.json", "w") as f:
    json.dump(explanations, f, indent=2)

# Save full shap matrix + base value for dashboard on-demand lookups
np.save(f"{OUT}/shap_values.npy", shap_values)
X_trans_df.assign(account_id=m["account_id"].values).to_parquet(f"{OUT}/shap_feature_matrix.parquet")
base_value = float(explainer.expected_value if np.isscalar(explainer.expected_value)
                   else np.array(explainer.expected_value).flatten()[0])
with open(f"{OUT}/reports/shap_base_value.json", "w") as f:
    json.dump({"base_value": base_value, "feature_names": feature_names.tolist()}, f)

print("SHAP explainability complete. Top drivers:")
print(mean_abs.head(10))
