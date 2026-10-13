# EDA Insights (Descriptive & Diagnostic Analytics)

- Overall: 500 customers, 110 churned, 390 retained, churn rate = 22.0%.
- Churn by initial plan tier: Enterprise=22.1%, Basic=22.0%, Pro=21.9%
- Churn by industry: DevTools=31.0%, FinTech=22.3%, HealthTech=21.9%, EdTech=16.5%, Cybersecurity=16.0%
- Churn by billing frequency: annual=19.6%, monthly=24.6%
- Churn by tenure bucket: 0-90d=32.7%, 91-180d=23.0%, 181-365d=27.8%, 1-2yr=12.0%
- Correlation with churn -> feature_adoption_rate: -0.48, usage_decline_rate: 0.05 (low adoption / rising usage decline both associate with higher churn).
- Avg satisfaction score - retained: 3.96, churned: 4.01.
- Top churn reasons (from churn_events, descriptive only): budget=27, features=27, support=27, pricing=23, competitor=18, unknown=13

## Top correlated features with churn

- distinct_features_used: -0.477
- feature_adoption_rate: -0.477
- total_usage_duration_secs: -0.422
- usage_event_count: -0.418
- total_usage_count: -0.417
- total_error_count: -0.395
- subscription_count: -0.369
- beta_feature_events: -0.242
- tenure_days: -0.207
- high_urgent_tickets: -0.180
