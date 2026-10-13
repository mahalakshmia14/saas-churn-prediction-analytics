"""
Phase 17-19: Revenue-at-risk, customer segmentation, and a documented
Customer Health Score. All weights are explicitly stated (no arbitrary
un-explained weighting), and Health Score is deliberately independent of
the ML model (behavioral heuristic, not a repackaged churn probability).
"""
import pandas as pd
import numpy as np
import os

OUT = "/home/claude/project/outputs"

m = pd.read_parquet(f"{OUT}/customer_master.parquet")
scored = pd.read_parquet(f"{OUT}/scored_customers.parquet")

df = scored.merge(
    m[["account_id", "feature_adoption_rate", "avg_satisfaction_score", "escalation_rate",
       "usage_decline_rate", "auto_renew_flag", "upgrade_rate", "downgrade_rate",
       "current_mrr", "has_satisfaction_data"]],
    on="account_id", suffixes=("", "_m")
)

# ---------------- Customer Health Score (0-100) ----------------
# Documented weighted blend of 5 behavioral pillars (weights sum to 100):
#  30% Feature adoption (engagement breadth)
#  20% Usage trend (1 - usage decline, capped 0-1)
#  20% Satisfaction (scaled 1-5 -> 0-1)
#  15% Support friction (inverse of escalation rate)
#  15% Commercial commitment (auto-renew + net upgrade behavior)
def clip01(s):
    return s.clip(0, 1)

adoption_score = clip01(df["feature_adoption_rate"])
usage_trend_score = clip01(1 - df["usage_decline_rate"])
satisfaction_score = clip01((df["avg_satisfaction_score"] - 1) / 4)
support_score = clip01(1 - df["escalation_rate"])
commitment_score = clip01(0.6 * df["auto_renew_flag"].astype(int) +
                           0.4 * clip01(df["upgrade_rate"] - df["downgrade_rate"] + 0.5))

df["health_score"] = (
    30 * adoption_score + 20 * usage_trend_score + 20 * satisfaction_score +
    15 * support_score + 15 * commitment_score
).round(1)

# ---------------- Segmentation: Value x Risk quadrant ----------------
mrr_median = df["current_mrr"].median()
risk_threshold = 0.5  # churn probability midpoint, documented business threshold


def segment(row):
    high_value = row["current_mrr"] >= mrr_median
    high_risk = row["churn_probability"] >= risk_threshold
    if high_value and high_risk:
        return "Priority Attention (High Value / High Risk)"
    if high_value and not high_risk:
        return "Protect & Expand (High Value / Low Risk)"
    if not high_value and high_risk:
        return "Automated Retention (Low Value / High Risk)"
    return "Normal Engagement (Low Value / Low Risk)"


df["segment"] = df.apply(segment, axis=1)

# ---------------- Revenue at Risk ----------------
# already have revenue_at_risk = churn_probability * current_mrr from train_model.py; recompute to be safe
df["revenue_at_risk"] = df["churn_probability"] * df["current_mrr"]

df.to_parquet(f"{OUT}/customer_analytics.parquet")
df.to_csv(f"{OUT}/reports/customer_analytics.csv", index=False)

print("Health score distribution:\n", df["health_score"].describe())
print("\nSegment counts:\n", df["segment"].value_counts())
print(f"\nTotal MRR at risk (expected value): ${df['revenue_at_risk'].sum():,.0f} / "
      f"Total current MRR: ${df['current_mrr'].sum():,.0f}")
