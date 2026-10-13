
# SaaS Customer Churn Prediction & Retention Analytics Platform

An end-to-end Data Analytics, Machine Learning, and Business Intelligence project built using the **RavenStack SaaS Subscription & Churn Analytics Dataset** (synthetic; credited to River @ Rivalytics on Kaggle).

The project analyzes subscription, product-usage, and support data to predict customer churn risk, explain model predictions, and provide actionable retention insights through an interactive Streamlit dashboard.

## 1. Features

- Customer churn prediction and risk assessment
- Exploratory data analysis and churn trend visualization
- Comparison of Logistic Regression, Random Forest, and XGBoost
- SHAP-based model explainability
- Customer health scoring and risk segmentation
- Revenue-at-risk analysis
- Rule-based retention recommendations
- Interactive what-if analysis

## 2. Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainability:** SHAP
- **Visualization:** Streamlit and the plotting libraries used in the project
- **Model Persistence:** Joblib

## 3. Project Structure

```text
saas-churn-project/
├── data/
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── train_model.py
│   ├── explain.py
│   ├── advanced_analytics.py
│   ├── recommendations.py
│   └── predict.py
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
├── app.py
├── requirements.txt
└── README.md
```

*This structure summarizes the main project components. Refer to the repository for the complete file structure.*

## 4. Installation & Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the analysis pipeline

Execute the scripts in the following order:

```bash
python src/data_processing.py
python src/feature_engineering.py
python src/eda.py
python src/train_model.py
python src/explain.py
python src/advanced_analytics.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

Run the pipeline first if the required model and output artifacts are not already available.

## 5. Data Processing & Feature Engineering

The project integrates account, subscription, feature-usage, support-ticket, and churn-event data into a customer-level dataset.

Key engineered features include:

- Feature adoption rate
- Usage decline rate
- Error rate
- Upgrade and downgrade rates
- Support ticket and escalation rates
- Customer tenure
- Current recurring revenue and subscription attributes

Implementation: `src/data_processing.py` and `src/feature_engineering.py`.

## 6. Target Definition & Leakage Prevention

The predictive target is `accounts.churn_flag`.

The project excludes outcome-related fields such as subscription churn flags, subscription end dates, and churn-event details from predictive features to reduce target leakage.

Behavioral features are calculated using account-specific reference dates. However, the dataset is a single synthetic snapshot rather than a true rolling time-series dataset. The reference-date approach is an approximation and does not replace temporal backtesting.

## 7. Machine Learning & Evaluation

Three classification models are compared using a stratified 80/20 train-test split with `random_state=42`:

- Logistic Regression
- Random Forest
- XGBoost

Evaluation metrics include accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrix, and ROC curve.

Logistic Regression achieved the highest ROC-AUC in the reported comparison and was selected for the saved model pipeline.

**Limitation:** Results from a synthetic dataset containing 500 accounts should not be interpreted as a benchmark of real-world churn prediction performance.

Results: `outputs/reports/model_results.json`.

## 8. Model Explainability

SHAP is used to explain model predictions and identify influential features.

The project generates global feature-importance visualizations and individual explanations for high-risk customers. These explanations are available in the dashboard.

SHAP values describe model associations; they do not establish causation or guarantee that changing a feature will prevent churn.

## 9. Advanced Analytics

- **Revenue at Risk:** Estimates exposure using churn probability multiplied by current monthly recurring revenue.
- **Customer Health Score:** Combines adoption, usage trends, satisfaction, support friction, and commercial indicators.
- **Customer Segmentation:** Groups customers by revenue value and predicted churn risk.
- **Retention Recommendations:** Suggests actions based on observed customer behavior using a rule-based approach.

These outputs support analysis and prioritization; they are not guarantees of business outcomes.

## 10. Streamlit Dashboard

The application includes seven pages:

1. **Executive Overview** — key performance indicators and risk distribution.
2. **Churn Analytics** — churn patterns across customer and subscription attributes.
3. **Customer Risk** — searchable and filterable customer risk table.
4. **Customer Details** — customer profiles, SHAP explanations, and recommendations.
5. **Model Performance** — model comparison and evaluation metrics.
6. **Revenue Risk** — revenue-at-risk analysis by customer segment.
7. **What-if Analysis** — interactive model-based scenario exploration.

## 11. Exploratory Data Analysis

The project examines churn rates and their associations with subscription plans, industry, billing frequency, tenure, product usage, support activity, and recorded churn reasons.

Detailed findings: `outputs/reports/eda_insights.md`.

## 12. Limitations

- The dataset is synthetic and contains 500 accounts.
- The target and churn-event records have documented inconsistencies.
- The reference-date approach is an approximation, not a rolling forecasting evaluation.
- Model performance may not generalize to real-world customer populations.
- SHAP explanations and retention recommendations do not establish causal effects.
- Real-world deployment would require representative data, temporal validation, and further operational testing.

## 13. Credits

**Dataset:** RavenStack SaaS Subscription & Churn Analytics Dataset, credited to River @ Rivalytics on Kaggle.

Refer to the original dataset listing for its applicable license and usage terms.
