"""
Phase 20: Rule-based retention recommendation engine.
These are SUGGESTED actions based on observed behavioral patterns,
not guaranteed solutions or causal proof.
"""

def get_recommendations(row):
    """row: a dict-like (pandas Series) with the customer's features."""
    recs = []

    low_usage = row.get("feature_adoption_rate", 1) < 0.25
    declining = row.get("usage_decline_rate", 0) > 0.2
    if low_usage or declining:
        recs.append({
            "trigger": "Low feature adoption / declining usage",
            "action": "Product training & onboarding",
            "detail": "Offer a guided onboarding session or in-app walkthrough to "
                       "increase feature adoption; usage decline often precedes churn."
        })

    high_tickets = row.get("total_tickets", 0) >= 5
    low_satisfaction = row.get("avg_satisfaction_score", 5) < 3
    high_escalation = row.get("escalation_rate", 0) > 0.3
    if high_tickets or low_satisfaction or high_escalation:
        recs.append({
            "trigger": "High support load / low satisfaction",
            "action": "Customer-success intervention",
            "detail": "Assign a dedicated CSM to review recent tickets and proactively "
                       "check in; repeated escalations and low CSAT correlate with churn."
        })

    downgraded = row.get("downgrade_rate", 0) > row.get("upgrade_rate", 0)
    high_risk = row.get("churn_probability", 0) >= 0.6
    if downgraded and high_risk:
        recs.append({
            "trigger": "Downgrade behavior + high churn risk",
            "action": "Review subscription fit",
            "detail": "Discuss plan/seat fit with the customer; a downgrade paired with "
                       "high predicted risk often signals a value or budget mismatch."
        })

    not_auto_renew = not bool(row.get("auto_renew_flag", True))
    if not_auto_renew and high_risk:
        recs.append({
            "trigger": "Auto-renew disabled + high churn risk",
            "action": "Billing / renewal follow-up",
            "detail": "Reach out ahead of the renewal date to confirm intent and address "
                       "any billing or budget concerns before the subscription lapses."
        })

    if not recs:
        recs.append({
            "trigger": "No strong risk signals detected",
            "action": "Normal engagement",
            "detail": "Continue standard engagement cadence; no elevated risk pattern found."
        })

    return recs
