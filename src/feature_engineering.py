"""
Phase 6 & 9: Aggregate the transactional tables to one row per customer
(account_id), using only records on/before that account's reference_date
(see data_processing.py docstring for the leakage-safe design).

Every engineered feature has a stated business interpretation (project
constraint: no features "just to add columns").
"""
import pandas as pd
import numpy as np
import os

OUT_DIR = "/home/claude/project/outputs"
TOTAL_FEATURES_IN_CATALOG = 40  # per dataset README: "pool of 40 SaaS features"


def build_master_dataset():
    accounts = pd.read_parquet(f"{OUT_DIR}/accounts_clean.parquet")
    subs = pd.read_parquet(f"{OUT_DIR}/subs_clean.parquet")
    usage = pd.read_parquet(f"{OUT_DIR}/usage_clean.parquet")
    tickets = pd.read_parquet(f"{OUT_DIR}/tickets_clean.parquet")
    ref = pd.read_parquet(f"{OUT_DIR}/reference_dates.parquet")

    base = accounts.merge(ref, on="account_id", how="left")

    # -------------------- ACCOUNT-LEVEL BASE FEATURES --------------------
    base["tenure_days"] = (base["reference_date"] - base["signup_date"]).dt.days.clip(lower=0)

    master = base[[
        "account_id", "industry", "country", "referral_source",
        "plan_tier", "seats", "is_trial", "tenure_days", "reference_date", "churn_flag"
    ]].rename(columns={"plan_tier": "initial_plan_tier", "seats": "initial_seats",
                        "is_trial": "account_is_trial"})

    # -------------------- SUBSCRIPTION FEATURES --------------------
    # Only subscriptions that had STARTED on/before the reference date are visible.
    subs_m = subs.merge(ref, on="account_id", how="left")
    subs_visible = subs_m[subs_m["start_date"] <= subs_m["reference_date"]].copy()

    # "Active as of reference date": started, and not yet ended (end_date null or in future)
    subs_visible["is_active_asof"] = (
        subs_visible["end_date"].isna() | (subs_visible["end_date"] > subs_visible["reference_date"])
    )

    sub_agg = subs_visible.groupby("account_id").agg(
        subscription_count=("subscription_id", "count"),
        total_upgrades=("upgrade_flag", "sum"),
        total_downgrades=("downgrade_flag", "sum"),
        max_seats_seen=("seats", "max"),
        annual_billing_ever=("billing_frequency", lambda s: (s == "annual").any()),
    ).reset_index()

    # Current (active) subscription snapshot: MRR/ARR/plan/billing/auto-renew "as of" reference date
    active_subs = subs_visible[subs_visible["is_active_asof"]].sort_values("start_date")
    current_snapshot = active_subs.groupby("account_id").agg(
        current_mrr=("mrr_amount", "last"),
        current_arr=("arr_amount", "last"),
        current_plan_tier=("plan_tier", "last"),
        current_billing_frequency=("billing_frequency", "last"),
        current_seats=("seats", "last"),
        auto_renew_flag=("auto_renew_flag", "last"),
        currently_trialing=("is_trial", "last"),
    ).reset_index()

    # For accounts with NO active subscription as of reference date (fully lapsed before churn
    # date derivation, edge case), fall back to their most recent subscription record.
    most_recent = subs_visible.sort_values("start_date").groupby("account_id").agg(
        current_mrr=("mrr_amount", "last"),
        current_arr=("arr_amount", "last"),
        current_plan_tier=("plan_tier", "last"),
        current_billing_frequency=("billing_frequency", "last"),
        current_seats=("seats", "last"),
        auto_renew_flag=("auto_renew_flag", "last"),
        currently_trialing=("is_trial", "last"),
    ).reset_index()

    current_snapshot = current_snapshot.set_index("account_id").combine_first(
        most_recent.set_index("account_id")
    ).reset_index()

    master = master.merge(sub_agg, on="account_id", how="left")
    master = master.merge(current_snapshot, on="account_id", how="left")

    master["subscription_count"] = master["subscription_count"].fillna(0)
    master["total_upgrades"] = master["total_upgrades"].fillna(0)
    master["total_downgrades"] = master["total_downgrades"].fillna(0)
    # Upgrade/Downgrade Behavior rate: normalized by subscription count (business feature)
    master["upgrade_rate"] = master["total_upgrades"] / master["subscription_count"].replace(0, np.nan)
    master["downgrade_rate"] = master["total_downgrades"] / master["subscription_count"].replace(0, np.nan)
    master["upgrade_rate"] = master["upgrade_rate"].fillna(0)
    master["downgrade_rate"] = master["downgrade_rate"].fillna(0)

    # -------------------- FEATURE USAGE FEATURES --------------------
    # Map subscription_id -> account_id (only visible subscriptions) so usage rows join to accounts
    sub_to_acct = subs_visible[["subscription_id", "account_id", "reference_date"]]
    usage_m = usage.merge(sub_to_acct, on="subscription_id", how="inner")
    usage_visible = usage_m[usage_m["usage_date"] <= usage_m["reference_date"]].copy()

    usage_overall = usage_visible.groupby("account_id").agg(
        total_usage_count=("usage_count", "sum"),
        total_usage_duration_secs=("usage_duration_secs", "sum"),
        total_error_count=("error_count", "sum"),
        distinct_features_used=("feature_name", "nunique"),
        beta_feature_events=("is_beta_feature", "sum"),
        usage_event_count=("usage_id", "count"),
    ).reset_index()

    # Feature Adoption = distinct features used / total feature catalog (40)
    usage_overall["feature_adoption_rate"] = (
        usage_overall["distinct_features_used"] / TOTAL_FEATURES_IN_CATALOG
    )
    # Error Rate = total errors / total usage
    usage_overall["error_rate"] = (
        usage_overall["total_error_count"] / usage_overall["total_usage_count"].replace(0, np.nan)
    ).fillna(0)
    # Beta adoption rate
    usage_overall["beta_feature_rate"] = (
        usage_overall["beta_feature_events"] / usage_overall["usage_event_count"].replace(0, np.nan)
    ).fillna(0)

    # Usage Decline: compare last-30-days usage vs prior-30-days usage before reference date
    usage_visible["days_before_ref"] = (usage_visible["reference_date"] - usage_visible["usage_date"]).dt.days
    recent = usage_visible[usage_visible["days_before_ref"] <= 30].groupby("account_id")["usage_count"].sum()
    prior = usage_visible[(usage_visible["days_before_ref"] > 30) & (usage_visible["days_before_ref"] <= 60)] \
        .groupby("account_id")["usage_count"].sum()
    decline_df = pd.DataFrame({"recent_30d_usage": recent, "prior_30d_usage": prior}).fillna(0).reset_index()
    decline_df["usage_decline_rate"] = np.where(
        decline_df["prior_30d_usage"] > 0,
        (decline_df["prior_30d_usage"] - decline_df["recent_30d_usage"]) / decline_df["prior_30d_usage"],
        0.0
    )

    master = master.merge(usage_overall, on="account_id", how="left")
    master = master.merge(decline_df[["account_id", "recent_30d_usage", "prior_30d_usage", "usage_decline_rate"]],
                           on="account_id", how="left")

    usage_cols_fillzero = ["total_usage_count", "total_usage_duration_secs", "total_error_count",
                            "distinct_features_used", "beta_feature_events", "usage_event_count",
                            "feature_adoption_rate", "error_rate", "beta_feature_rate",
                            "recent_30d_usage", "prior_30d_usage", "usage_decline_rate"]
    for c in usage_cols_fillzero:
        master[c] = master[c].fillna(0)

    # Usage rate normalized by tenure (avoid pure-volume bias for very new accounts)
    master["usage_per_tenure_day"] = master["total_usage_count"] / master["tenure_days"].replace(0, np.nan)
    master["usage_per_tenure_day"] = master["usage_per_tenure_day"].fillna(0)

    # -------------------- SUPPORT TICKET FEATURES --------------------
    tickets_m = tickets.merge(ref, on="account_id", how="left")
    tickets_visible = tickets_m[tickets_m["submitted_at"] <= tickets_m["reference_date"]].copy()

    ticket_agg = tickets_visible.groupby("account_id").agg(
        total_tickets=("ticket_id", "count"),
        avg_resolution_time_hours=("resolution_time_hours", "mean"),
        avg_first_response_minutes=("first_response_time_minutes", "mean"),
        avg_satisfaction_score=("satisfaction_score", "mean"),
        pct_missing_satisfaction=("satisfaction_score", lambda s: s.isna().mean()),
        escalation_count=("escalation_flag", "sum"),
        high_urgent_tickets=("priority", lambda s: s.isin(["high", "urgent"]).sum()),
    ).reset_index()

    master = master.merge(ticket_agg, on="account_id", how="left")
    ticket_cols_fillzero = ["total_tickets", "escalation_count", "high_urgent_tickets"]
    for c in ticket_cols_fillzero:
        master[c] = master[c].fillna(0)
    master["avg_resolution_time_hours"] = master["avg_resolution_time_hours"].fillna(
        master["avg_resolution_time_hours"].median())
    master["avg_first_response_minutes"] = master["avg_first_response_minutes"].fillna(
        master["avg_first_response_minutes"].median())
    master["pct_missing_satisfaction"] = master["pct_missing_satisfaction"].fillna(1.0)
    # avg_satisfaction_score: leave NaN where no tickets/no responses -> impute with median, flag separately
    master["has_satisfaction_data"] = master["avg_satisfaction_score"].notna().astype(int)
    master["avg_satisfaction_score"] = master["avg_satisfaction_score"].fillna(
        master["avg_satisfaction_score"].median())

    # Support Ticket Rate = tickets / tenure period (per 30 days of tenure, business-friendly units)
    master["tickets_per_30_tenure_days"] = (
        master["total_tickets"] / (master["tenure_days"].replace(0, np.nan) / 30)
    ).fillna(0)
    master["escalation_rate"] = (
        master["escalation_count"] / master["total_tickets"].replace(0, np.nan)
    ).fillna(0)

    # -------------------- FINAL CLEANUP --------------------
    master["churn"] = master["churn_flag"].astype(int)
    master = master.drop(columns=["churn_flag"])

    # current_plan_tier / billing / seats fallback to account-level initial values if still missing
    master["current_plan_tier"] = master["current_plan_tier"].fillna(master["initial_plan_tier"])
    master["current_seats"] = master["current_seats"].fillna(master["initial_seats"])
    master["current_billing_frequency"] = master["current_billing_frequency"].fillna("monthly")
    master["current_mrr"] = master["current_mrr"].fillna(0)
    master["current_arr"] = master["current_arr"].fillna(0)
    master["auto_renew_flag"] = master["auto_renew_flag"].fillna(False)
    master["currently_trialing"] = master["currently_trialing"].fillna(False)
    master["annual_billing_ever"] = master["annual_billing_ever"].fillna(False)
    master["max_seats_seen"] = master["max_seats_seen"].fillna(master["initial_seats"])

    return master


if __name__ == "__main__":
    master = build_master_dataset()
    print("Master dataset shape:", master.shape)
    print("Rows = unique accounts:", master["account_id"].nunique() == len(master))
    print("Churn rate:", master["churn"].mean().round(4))
    print(master.dtypes)
    master.to_parquet(f"{OUT_DIR}/customer_master.parquet")
    master.to_csv(f"{OUT_DIR}/reports/customer_master.csv", index=False)
    print("Saved customer_master.parquet/.csv")
