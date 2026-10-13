"""
SaaS Customer Churn Prediction & Retention Analytics Platform
Streamlit Dashboard - app.py

Run with:  streamlit run app.py
(from the project root, after running the src/ pipeline scripts once to
generate outputs/ and models/)
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from recommendations import get_recommendations  # noqa: E402

st.set_page_config(page_title="RavenStack Churn Analytics", layout="wide", page_icon="📉")

OUT = os.path.join(os.path.dirname(__file__), "outputs")
MODELS = os.path.join(os.path.dirname(__file__), "models")


# ---------------------------------------------------------------- #
# Data loading (cached)
# ---------------------------------------------------------------- #
@st.cache_data
def load_data():
    analytics = pd.read_parquet(f"{OUT}/customer_analytics.parquet")
    master = pd.read_parquet(f"{OUT}/customer_master.parquet")
    with open(f"{OUT}/reports/model_results.json") as f:
        model_results = json.load(f)
    with open(f"{OUT}/reports/eda_insights.md") as f:
        eda_text = f.read()
    shap_top = pd.read_csv(f"{OUT}/reports/shap_top_features.csv", index_col=0)
    with open(f"{OUT}/reports/shap_explanations.json") as f:
        shap_explanations = json.load(f)
    return analytics, master, model_results, eda_text, shap_top, shap_explanations


@st.cache_resource
def load_model():
    pipeline = joblib.load(f"{MODELS}/trained_model.pkl")
    meta = joblib.load(f"{MODELS}/model_metadata.pkl")
    return pipeline, meta


analytics, master, model_results, eda_text, shap_top, shap_explanations = load_data()
pipeline, meta = load_model()

# merge analytics with master for full feature access on customer detail page
full = analytics.merge(
    master.drop(columns=["churn"]), on="account_id", how="left", suffixes=("", "_full")
)

st.sidebar.title("📉 RavenStack Churn Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Executive Overview", "Churn Analytics", "Customer Risk", "Customer Details",
     "Model Performance", "Revenue Risk", "What-if Analysis"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Synthetic dataset (RavenStack, via Kaggle). Model performance reflects "
    "synthetic data and should not be read as real-world production accuracy."
)

RISK_COLORS = {"Low": "#4C956C", "Medium": "#F2C94C", "High": "#F2994A", "Critical": "#EB5757"}

# ============================================================== #
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================== #
if page == "Executive Overview":
    st.title("Executive Overview")

    total = len(analytics)
    churned = int(analytics["churn"].sum())
    churn_rate = churned / total
    high_risk = int((analytics["risk_level"].isin(["High", "Critical"])).sum())
    avg_prob = analytics["churn_probability"].mean()
    revenue_at_risk = analytics["revenue_at_risk"].sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Overall Churn Rate", f"{churn_rate:.1%}")
    c4.metric("High/Critical Risk", f"{high_risk:,}")
    c5.metric("Avg. Churn Probability", f"{avg_prob:.1%}")
    c6.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Risk Level Distribution")
        risk_counts = analytics["risk_level"].value_counts().reindex(["Low", "Medium", "High", "Critical"])
        fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                     color=risk_counts.index, color_discrete_map=RISK_COLORS,
                     labels={"x": "Risk Level", "y": "Customers"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Customer Segments")
        seg_counts = analytics["segment"].value_counts()
        fig2 = px.pie(names=seg_counts.index, values=seg_counts.values, hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Insights")
    for line in eda_text.split("\n"):
        if line.startswith("- "):
            st.markdown(line)

# ============================================================== #
# PAGE 2 — CHURN ANALYTICS
# ============================================================== #
elif page == "Churn Analytics":
    st.title("Churn Analytics")
    st.caption("Descriptive & diagnostic analysis answering: who churns, and what's associated with it?")

    tab1, tab2, tab3, tab4 = st.tabs(["By Segment", "By Tenure & Usage", "By Support", "Churn Reasons"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            plan = master.groupby("initial_plan_tier")["churn"].mean().reset_index()
            fig = px.bar(plan, x="initial_plan_tier", y="churn", title="Churn Rate by Plan Tier",
                         labels={"churn": "Churn Rate"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ind = master.groupby("industry")["churn"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(ind, x="churn", y="industry", orientation="h", title="Churn Rate by Industry",
                         labels={"churn": "Churn Rate"})
            st.plotly_chart(fig, use_container_width=True)
        bill = master.groupby("current_billing_frequency")["churn"].mean().reset_index()
        fig = px.bar(bill, x="current_billing_frequency", y="churn", title="Churn Rate by Billing Frequency",
                     labels={"churn": "Churn Rate"})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        m2 = master.copy()
        m2["tenure_bucket"] = pd.cut(m2["tenure_days"], bins=[-1, 90, 180, 365, 730, 10000],
                                      labels=["0-90d", "91-180d", "181-365d", "1-2yr", "2yr+"])
        tenure = m2.groupby("tenure_bucket", observed=True)["churn"].mean().reset_index()
        fig = px.bar(tenure, x="tenure_bucket", y="churn", title="Churn Rate by Tenure",
                     labels={"churn": "Churn Rate"})
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(master, x="churn", y="feature_adoption_rate", color="churn",
                         title="Feature Adoption vs Churn",
                         labels={"churn": "Churn (0=No, 1=Yes)"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(master, x="churn", y="usage_decline_rate", color="churn",
                         title="Usage Decline Rate vs Churn",
                         labels={"churn": "Churn (0=No, 1=Yes)"})
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(master, x="churn", y="total_tickets", color="churn",
                         title="Support Tickets vs Churn")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(master, x="churn", y="avg_satisfaction_score", color="churn",
                         title="Avg Satisfaction Score vs Churn")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.info("Churn reasons come from post-outcome churn_events data — shown for descriptive "
                "context only, never used as a predictive feature.")
        reason_col = "reason_code" if "reason_code" in master.columns else None
        try:
            churn_events = pd.read_parquet(f"{OUT}/churn_events_clean.parquet")
            churned_ids = set(master.loc[master["churn"] == 1, "account_id"])
            ce = churn_events[churn_events["account_id"].isin(churned_ids)]
            reasons = ce["reason_code"].value_counts().reset_index()
            reasons.columns = ["reason_code", "count"]
            fig = px.bar(reasons, x="count", y="reason_code", orientation="h",
                         title="Churn Reasons (descriptive)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load churn events: {e}")

# ============================================================== #
# PAGE 3 — CUSTOMER RISK
# ============================================================== #
elif page == "Customer Risk":
    st.title("Customer Risk Table")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_filter = st.multiselect("Risk Level", ["Low", "Medium", "High", "Critical"],
                                      default=["High", "Critical"])
    with col2:
        plan_filter = st.multiselect("Plan", sorted(analytics["current_plan_tier"].unique()),
                                      default=list(sorted(analytics["current_plan_tier"].unique())))
    with col3:
        industry_filter = st.multiselect("Industry", sorted(analytics["industry"].unique()),
                                          default=list(sorted(analytics["industry"].unique())))
    with col4:
        search = st.text_input("Search Customer ID")

    filtered = analytics[
        analytics["risk_level"].isin(risk_filter) &
        analytics["current_plan_tier"].isin(plan_filter) &
        analytics["industry"].isin(industry_filter)
    ]
    if search:
        filtered = filtered[filtered["account_id"].str.contains(search, case=False)]

    filtered = filtered.sort_values("churn_probability", ascending=False)
    display_cols = ["account_id", "current_plan_tier", "current_mrr", "churn_probability",
                     "risk_level", "total_tickets", "feature_adoption_rate", "health_score", "segment"]
    show = filtered[display_cols].rename(columns={
        "account_id": "Customer ID", "current_plan_tier": "Plan", "current_mrr": "MRR",
        "churn_probability": "Churn Probability", "risk_level": "Risk Level",
        "total_tickets": "Support Tickets", "feature_adoption_rate": "Feature Adoption",
        "health_score": "Health Score", "segment": "Segment"
    })
    show["Churn Probability"] = (show["Churn Probability"] * 100).round(1).astype(str) + "%"
    show["MRR"] = show["MRR"].apply(lambda x: f"${x:,.0f}")
    show["Feature Adoption"] = (show["Feature Adoption"] * 100).round(1).astype(str) + "%"

    st.caption(f"{len(filtered)} customers match the current filters.")
    st.dataframe(show, use_container_width=True, height=500)

# ============================================================== #
# PAGE 4 — CUSTOMER DETAILS
# ============================================================== #
elif page == "Customer Details":
    st.title("Customer Details")

    acct = st.selectbox("Select Customer ID",
                         sorted(analytics["account_id"]),
                         index=int(analytics["churn_probability"].values.argmax()))

    row = full[full["account_id"] == acct].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn Probability", f"{row['churn_probability']:.1%}")
    c2.metric("Risk Level", row["risk_level"])
    c3.metric("Health Score", f"{row['health_score']:.0f}/100")
    c4.metric("Monthly Revenue at Risk", f"${row['revenue_at_risk']:,.0f}")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Account")
        st.write(f"**Industry:** {row['industry']}")
        st.write(f"**Country:** {row['country']}")
        st.write(f"**Referral Source:** {row['referral_source']}")
        st.write(f"**Tenure (days):** {row['tenure_days']:.0f}")
        st.write(f"**Segment:** {row['segment']}")
    with col2:
        st.subheader("Subscription")
        st.write(f"**Plan:** {row['current_plan_tier']}")
        st.write(f"**MRR:** ${row['current_mrr']:,.0f}")
        st.write(f"**Seats:** {row['current_seats']:.0f}")
        st.write(f"**Billing:** {row['current_billing_frequency']}")
        st.write(f"**Auto-renew:** {bool(row['auto_renew_flag'])}")
    with col3:
        st.subheader("Usage & Support")
        st.write(f"**Feature adoption:** {row['feature_adoption_rate']:.1%}")
        st.write(f"**Usage decline rate:** {row['usage_decline_rate']:.1%}")
        st.write(f"**Support tickets:** {row['total_tickets']:.0f}")
        st.write(f"**Avg satisfaction:** {row['avg_satisfaction_score']:.1f}/5")
        st.write(f"**Escalation rate:** {row['escalation_rate']:.1%}")

    st.markdown("---")
    st.subheader("Why is this customer at risk? (SHAP)")
    if acct in shap_explanations:
        exp = shap_explanations[acct]
        exp_df = pd.DataFrame(exp)
        exp_df["direction"] = np.where(exp_df["shap_value"] > 0, "↑ increases risk", "↓ decreases risk")
        fig = px.bar(exp_df, x="shap_value", y="feature", orientation="h", color="direction",
                     color_discrete_map={"↑ increases risk": "#EB5757", "↓ decreases risk": "#4C956C"},
                     title="Top contributing factors")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("SHAP values show model associations, not proven causal reasons. They explain "
                   "*this model's* prediction, not a guarantee about real-world cause and effect.")
    else:
        st.info("Detailed SHAP breakdown is precomputed for the top 50 highest-risk customers. "
                "This customer is not in that set (lower relative risk).")

    st.markdown("---")
    st.subheader("Suggested Retention Actions")
    for rec in get_recommendations(row):
        with st.expander(f"🎯 {rec['action']}  —  *trigger: {rec['trigger']}*"):
            st.write(rec["detail"])
    st.caption("These are rule-based suggestions, not guaranteed solutions.")


# PAGE 5 — MODEL PERFORMANCE

elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Reflects performance on a held-out test split of the synthetic RavenStack dataset. "
               "Not representative of real-world production accuracy.")

    best_model = model_results["best_model"]
    st.success(f"Best model selected objectively by ROC-AUC on the test set: **{best_model}**")

    metric_rows = []
    for name, r in model_results["results"].items():
        row = {"Model": name}
        row.update({k: round(v, 3) for k, v in r["metrics"].items()})
        metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows).set_index("Model")
    st.subheader("Model Comparison")
    st.dataframe(metrics_df, use_container_width=True)

    fig = px.bar(metrics_df.reset_index().melt(id_vars="Model"), x="Model", y="value", color="variable",
                 barmode="group", title="Metric Comparison Across Models")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"ROC Curve — {best_model}")
        roc = model_results["results"][best_model]["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=best_model))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader(f"Confusion Matrix — {best_model}")
        cm = np.array(model_results["results"][best_model]["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, x=["Pred: No Churn", "Pred: Churn"],
                         y=["Actual: No Churn", "Actual: Churn"], color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Churn Risk Drivers (SHAP, global)")
    st.bar_chart(shap_top["mean_abs_shap"].head(15))


# PAGE 6 — REVENUE RISK

elif page == "Revenue Risk":
    st.title("Revenue Risk")
    st.caption("Revenue at Risk = Churn Probability × MRR — an expected-value style exposure metric, "
               "not a guaranteed revenue loss.")

    total_mrr = analytics["current_mrr"].sum()
    revenue_at_risk = analytics["revenue_at_risk"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Current MRR", f"${total_mrr:,.0f}")
    c2.metric("Total Revenue at Risk", f"${revenue_at_risk:,.0f}")
    c3.metric("% of MRR at Risk", f"{revenue_at_risk/total_mrr:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        by_plan = analytics.groupby("current_plan_tier")["revenue_at_risk"].sum().reset_index()
        fig = px.bar(by_plan, x="current_plan_tier", y="revenue_at_risk", title="Revenue at Risk by Plan")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        by_seg = analytics.groupby("segment")["revenue_at_risk"].sum().reset_index()
        fig = px.bar(by_seg, x="revenue_at_risk", y="segment", orientation="h",
                     title="Revenue at Risk by Segment")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 Accounts by Revenue at Risk")
    top15 = analytics.sort_values("revenue_at_risk", ascending=False).head(15)[
        ["account_id", "current_plan_tier", "current_mrr", "churn_probability", "revenue_at_risk"]
    ]
    st.dataframe(top15, use_container_width=True)

# PAGE 7 — WHAT-IF ANALYSIS

elif page == "What-if Analysis":
    st.title("What-if Scenario Analysis")
    st.info("This is model-based scenario analysis, not causal proof that these changes will "
            "reduce a real customer's churn risk.")

    acct = st.selectbox("Base customer", sorted(analytics["account_id"]))
    base_row = full[full["account_id"] == acct].iloc[0].copy()

    st.subheader("Adjust behavior")
    col1, col2, col3 = st.columns(3)
    with col1:
        adoption = st.slider("Feature adoption rate", 0.0, 1.0, float(base_row["feature_adoption_rate"]))
    with col2:
        decline = st.slider("Usage decline rate", 0.0, 1.0, float(base_row["usage_decline_rate"]))
    with col3:
        tickets = st.slider("Support tickets", 0, 30, int(base_row["total_tickets"]))

    scenario_row = base_row.copy()
    scenario_row["feature_adoption_rate"] = adoption
    scenario_row["usage_decline_rate"] = decline
    scenario_row["total_tickets"] = tickets

    def predict_one(row):
        X = pd.DataFrame([row[meta["features"]]])
        for c in meta["boolean"]:
            X[c] = X[c].astype(int)
        return float(pipeline.predict_proba(X)[:, 1][0])

    base_pred = predict_one(base_row)
    scenario_pred = predict_one(scenario_row)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Predicted Churn", f"{base_pred:.1%}")
    c2.metric("Scenario Predicted Churn", f"{scenario_pred:.1%}", delta=f"{(scenario_pred-base_pred):+.1%}")
    c3.metric("Direction", "⬇ Improved" if scenario_pred < base_pred else
              ("⬆ Worsened" if scenario_pred > base_pred else "No change"))

    fig = go.Figure(go.Bar(x=["Current", "Scenario"], y=[base_pred, scenario_pred],
                            marker_color=["#F2994A", "#4C956C"]))
    fig.update_layout(yaxis_title="Predicted Churn Probability", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
