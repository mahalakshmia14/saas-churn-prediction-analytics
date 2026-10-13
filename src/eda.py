import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
OUT = "/home/claude/project/outputs"
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

m = pd.read_parquet(f"{OUT}/customer_master.parquet")
churn_events = pd.read_parquet(f"{OUT}/churn_events_clean.parquet")
accounts = pd.read_parquet(f"{OUT}/accounts_clean.parquet")

insights = []

# 1. Overall churn
total = len(m); churned = m["churn"].sum(); rate = churned / total
insights.append(f"Overall: {total} customers, {churned} churned, {total-churned} retained, "
                 f"churn rate = {rate:.1%}.")

plt.figure(figsize=(4, 4))
m["churn"].map({0: "Retained", 1: "Churned"}).value_counts().plot.pie(
    autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"])
plt.ylabel("")
plt.title("Overall Churn Split")
plt.tight_layout()
plt.savefig(f"{FIG}/01_overall_churn.png", dpi=120)
plt.close()

# 2. Churn by plan
plan_churn = m.groupby("initial_plan_tier")["churn"].mean().sort_values(ascending=False)
insights.append("Churn by initial plan tier: " + ", ".join(f"{k}={v:.1%}" for k, v in plan_churn.items()))
plt.figure(figsize=(5, 4))
sns.barplot(x=plan_churn.index, y=plan_churn.values, hue=plan_churn.index, palette="Blues_d", legend=False)
plt.ylabel("Churn rate"); plt.title("Churn Rate by Plan Tier")
plt.tight_layout(); plt.savefig(f"{FIG}/02_churn_by_plan.png", dpi=120); plt.close()

# 3. Churn by industry
ind_churn = m.groupby("industry")["churn"].mean().sort_values(ascending=False)
insights.append("Churn by industry: " + ", ".join(f"{k}={v:.1%}" for k, v in ind_churn.items()))
plt.figure(figsize=(6, 4))
sns.barplot(x=ind_churn.values, y=ind_churn.index, hue=ind_churn.index, palette="Oranges_d", legend=False)
plt.xlabel("Churn rate"); plt.title("Churn Rate by Industry")
plt.tight_layout(); plt.savefig(f"{FIG}/03_churn_by_industry.png", dpi=120); plt.close()

# 4. Churn by billing frequency
bill_churn = m.groupby("current_billing_frequency")["churn"].mean()
insights.append("Churn by billing frequency: " + ", ".join(f"{k}={v:.1%}" for k, v in bill_churn.items()))
plt.figure(figsize=(4, 4))
sns.barplot(x=bill_churn.index, y=bill_churn.values, hue=bill_churn.index, palette="Greens_d", legend=False)
plt.ylabel("Churn rate"); plt.title("Churn Rate by Billing Frequency")
plt.tight_layout(); plt.savefig(f"{FIG}/04_churn_by_billing.png", dpi=120); plt.close()

# 5. Churn by tenure
m["tenure_bucket"] = pd.cut(m["tenure_days"], bins=[-1, 90, 180, 365, 730, 10000],
                             labels=["0-90d", "91-180d", "181-365d", "1-2yr", "2yr+"])
tenure_churn = m.groupby("tenure_bucket", observed=True)["churn"].mean()
insights.append("Churn by tenure bucket: " + ", ".join(f"{k}={v:.1%}" for k, v in tenure_churn.items()))
plt.figure(figsize=(5, 4))
sns.barplot(x=tenure_churn.index, y=tenure_churn.values, hue=tenure_churn.index, palette="Purples_d", legend=False)
plt.ylabel("Churn rate"); plt.title("Churn Rate by Tenure"); plt.xticks(rotation=20)
plt.tight_layout(); plt.savefig(f"{FIG}/05_churn_by_tenure.png", dpi=120); plt.close()

# 6. Churn by usage (feature adoption + usage decline)
plt.figure(figsize=(5, 4))
sns.boxplot(x="churn", y="feature_adoption_rate", hue="churn", data=m, palette="coolwarm", legend=False)
plt.xticks([0, 1], ["Retained", "Churned"]); plt.title("Feature Adoption vs Churn")
plt.tight_layout(); plt.savefig(f"{FIG}/06_adoption_vs_churn.png", dpi=120); plt.close()

adopt_corr = m[["feature_adoption_rate", "usage_decline_rate", "churn"]].corr()["churn"]
insights.append(f"Correlation with churn -> feature_adoption_rate: {adopt_corr['feature_adoption_rate']:.2f}, "
                 f"usage_decline_rate: {adopt_corr['usage_decline_rate']:.2f} "
                 f"(low adoption / rising usage decline both associate with higher churn).")

# 7. Churn by support behavior
plt.figure(figsize=(5, 4))
sns.boxplot(x="churn", y="total_tickets", hue="churn", data=m, palette="coolwarm", legend=False)
plt.xticks([0, 1], ["Retained", "Churned"]); plt.title("Support Tickets vs Churn")
plt.tight_layout(); plt.savefig(f"{FIG}/07_tickets_vs_churn.png", dpi=120); plt.close()

sat_churn = m.groupby("churn")["avg_satisfaction_score"].mean()
insights.append(f"Avg satisfaction score - retained: {sat_churn.get(0, float('nan')):.2f}, "
                 f"churned: {sat_churn.get(1, float('nan')):.2f}.")

# 8. Churn reasons (post-outcome descriptive only, from churn_events)
churned_ids = set(m.loc[m["churn"] == 1, "account_id"])
ce_for_churned = churn_events[churn_events["account_id"].isin(churned_ids)]
reason_counts = ce_for_churned["reason_code"].value_counts()
insights.append("Top churn reasons (from churn_events, descriptive only): " +
                 ", ".join(f"{k}={v}" for k, v in reason_counts.head(6).items()))
plt.figure(figsize=(6, 4))
sns.barplot(x=reason_counts.values, y=reason_counts.index, hue=reason_counts.index, palette="Reds_d", legend=False)
plt.xlabel("Count"); plt.title("Churn Reasons (post-outcome, descriptive)")
plt.tight_layout(); plt.savefig(f"{FIG}/08_churn_reasons.png", dpi=120); plt.close()

# Correlation heatmap of numeric features vs churn
numeric_cols = m.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "churn"]
corr_with_churn = m[numeric_cols + ["churn"]].corr()["churn"].drop("churn").sort_values(key=abs, ascending=False)
plt.figure(figsize=(6, 8))
sns.barplot(x=corr_with_churn.values, y=corr_with_churn.index, hue=corr_with_churn.index,
            palette="vlag", legend=False)
plt.title("Feature Correlation with Churn"); plt.xlabel("Correlation")
plt.tight_layout(); plt.savefig(f"{FIG}/09_correlation_with_churn.png", dpi=120); plt.close()

with open(f"{OUT}/reports/eda_insights.md", "w") as f:
    f.write("# EDA Insights (Descriptive & Diagnostic Analytics)\n\n")
    for i in insights:
        f.write(f"- {i}\n")
    f.write("\n## Top correlated features with churn\n\n")
    for k, v in corr_with_churn.head(10).items():
        f.write(f"- {k}: {v:.3f}\n")

print("EDA complete.")
for i in insights:
    print("-", i)
