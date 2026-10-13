"""
Phases 2-6: Load & clean data, understand relationships, design a leakage-safe
prediction target, and build the customer-level master dataset.

KEY DESIGN DECISION (documented limitation - see README):
The RavenStack dataset is a single synthetic snapshot, not a rolling multi-period
event stream with a clean "prediction-as-of" timestamp supplied by the source.
To avoid target/temporal leakage we construct one ourselves:

  - TARGET: accounts.churn_flag ("churned at any point") is used as the label.
    This is the only fully consistent churn indicator in the dataset - the
    churn_events table is inconsistent with accounts.churn_flag for a sizable
    number of accounts (likely synthetic noise / reactivation edge cases), so
    churn_events is used ONLY for descriptive/diagnostic analysis of *why*
    churned customers left, never as a predictive feature or as the label.

  - REFERENCE DATE ("as of" date) per account:
      * Churned accounts: the earliest end_date among that account's
        subscriptions flagged churn_flag=True (falls back to the account's
        earliest churn_events churn_date if no such subscription exists).
      * Active accounts: the global dataset snapshot date (max date seen
        across all tables), simulating "today".

  - All behavioral features (usage, tickets, subscription activity) are
    aggregated using ONLY records dated on/before an account's reference date.
    Nothing dated after the reference date is ever used as a feature.
    subscriptions.churn_flag and subscriptions.end_date are NEVER used as
    features (they are the leakage columns), only as the (one-time) source
    for computing a churned account's reference date.

This is a simplified, documented approximation of a real "predict future
churn from past behavior" setup - not a true rolling forecast window. See
README.md Limitations section.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = "/home/claude/project/data"
OUT_DIR = "/home/claude/project/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42

# ----------------------------------------------------------------------
# PHASE 2: Load & clean all five tables
# ----------------------------------------------------------------------

def load_data():
    accounts = pd.read_csv(f"{DATA_DIR}/ravenstack_accounts.csv")
    subs = pd.read_csv(f"{DATA_DIR}/ravenstack_subscriptions.csv")
    usage = pd.read_csv(f"{DATA_DIR}/ravenstack_feature_usage.csv")
    tickets = pd.read_csv(f"{DATA_DIR}/ravenstack_support_tickets.csv")
    churn_events = pd.read_csv(f"{DATA_DIR}/ravenstack_churn_events.csv")

    # Parse dates
    accounts["signup_date"] = pd.to_datetime(accounts["signup_date"])
    subs["start_date"] = pd.to_datetime(subs["start_date"])
    subs["end_date"] = pd.to_datetime(subs["end_date"])
    usage["usage_date"] = pd.to_datetime(usage["usage_date"])
    tickets["submitted_at"] = pd.to_datetime(tickets["submitted_at"])
    tickets["closed_at"] = pd.to_datetime(tickets["closed_at"])
    churn_events["churn_date"] = pd.to_datetime(churn_events["churn_date"])

    # Data quality checks
    for name, df in [("accounts", accounts), ("subscriptions", subs),
                      ("feature_usage", usage), ("support_tickets", tickets),
                      ("churn_events", churn_events)]:
        dupes = df.duplicated().sum()
        if dupes:
            print(f"[WARN] {name}: {dupes} exact duplicate rows found - dropping")
            df.drop_duplicates(inplace=True)

    # Primary key duplicate checks
    assert accounts["account_id"].is_unique, "accounts.account_id not unique!"
    assert subs["subscription_id"].is_unique, "subscriptions.subscription_id not unique!"
    if not usage["usage_id"].is_unique:
        n_dupe_ids = usage["usage_id"].duplicated().sum()
        print(f"[WARN] feature_usage.usage_id has {n_dupe_ids} colliding IDs "
              f"(rows themselves are not duplicates - likely random ID generation "
              f"collision in the synthetic generator). Replacing with a surrogate "
              f"row-level key; no rows dropped.")
        usage = usage.reset_index(drop=True)
        usage["usage_row_id"] = "URW-" + usage.index.astype(str)
    assert tickets["ticket_id"].is_unique, "support_tickets.ticket_id not unique!"

    return accounts, subs, usage, tickets, churn_events


# ----------------------------------------------------------------------
# PHASE 5: Design a leakage-safe target / reference date
# ----------------------------------------------------------------------

def compute_reference_dates(accounts, subs, churn_events):
    snapshot_date = max(
        subs["start_date"].max(), subs["end_date"].max(skipna=True),
        churn_events["churn_date"].max(), accounts["signup_date"].max()
    )

    # earliest end_date among churned subscriptions, per account
    churned_subs = subs[subs["churn_flag"] == True]
    sub_churn_date = churned_subs.groupby("account_id")["end_date"].min()

    # fallback: earliest churn_events date, per account
    ce_churn_date = churn_events.groupby("account_id")["churn_date"].min()

    ref = accounts[["account_id", "churn_flag", "signup_date"]].copy()
    ref["reference_date"] = ref["account_id"].map(sub_churn_date)
    missing = ref["reference_date"].isna() & ref["churn_flag"]
    ref.loc[missing, "reference_date"] = ref.loc[missing, "account_id"].map(ce_churn_date)

    # any churned account still missing a reference date (edge case) -> snapshot date
    still_missing = ref["reference_date"].isna() & ref["churn_flag"]
    n_fallback = still_missing.sum()
    if n_fallback:
        print(f"[INFO] {n_fallback} churned accounts had no derivable churn date; "
              f"using snapshot date as fallback reference date.")
    ref.loc[still_missing, "reference_date"] = snapshot_date

    # active accounts: snapshot date
    ref.loc[~ref["churn_flag"], "reference_date"] = snapshot_date

    # guard: reference date can never be before signup
    ref["reference_date"] = ref[["reference_date", "signup_date"]].max(axis=1)

    return ref[["account_id", "reference_date"]], snapshot_date


if __name__ == "__main__":
    accounts, subs, usage, tickets, churn_events = load_data()
    print("Loaded shapes:", accounts.shape, subs.shape, usage.shape, tickets.shape, churn_events.shape)

    ref_dates, snapshot_date = compute_reference_dates(accounts, subs, churn_events)
    print("Snapshot (global 'today') date:", snapshot_date)
    print(ref_dates.head())

    ref_dates.to_parquet(f"{OUT_DIR}/reference_dates.parquet")
    accounts.to_parquet(f"{OUT_DIR}/accounts_clean.parquet")
    subs.to_parquet(f"{OUT_DIR}/subs_clean.parquet")
    usage.to_parquet(f"{OUT_DIR}/usage_clean.parquet")
    tickets.to_parquet(f"{OUT_DIR}/tickets_clean.parquet")
    churn_events.to_parquet(f"{OUT_DIR}/churn_events_clean.parquet")
    print("Phase 2-5 complete. Cleaned tables + reference dates saved.")
