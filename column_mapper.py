import re


COLUMN_ALIASES = {
    "customer_id": ["customerid", "customer_id", "customer id", "id", "account_id"],
    "churn": ["churn", "churn_flag", "churned", "is_churned", "attrition", "exited"],
    "monthly_charges": ["monthly_charges", "monthlycharges", "monthly charge", "monthly_charge"],
    "contract": ["contract", "contract_type", "contracttype"],
    "plan": ["plan", "plan_type", "plantype", "service_plan"],
    "subscription": ["subscription_type", "subscriptiontype", "subscription"],
    "customer_value": ["cltv", "customer_lifetime_value", "totalcharges", "total_charges"],
    "churn_score": ["churn_score", "churnscore", "risk_score"],
    "complaints": ["complaint_count", "complaints", "complaint"],
    "csat": ["csat_score", "csat", "customer_satisfaction"],
    "escalations": ["escalations", "escalation_count"],
    "cancellation_reason": ["cancellation_reason", "cancel_reason", "reason"],
    "cancellation_date": ["cancellation_date", "cancel_date", "date_cancelled"],
    "gender": ["gender", "sex"],
    "state": ["state", "region", "province", "country"],
}


def normalize_column(column):
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def detect_columns(df):
    normalized = {}
    for column in df.columns:
        normalized.setdefault(normalize_column(column), column)

    mapping = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            match = normalized.get(normalize_column(alias))
            if match is not None:
                mapping[standard_name] = match
                break
    return mapping
