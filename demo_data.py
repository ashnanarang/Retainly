import random
from datetime import date, timedelta
from math import exp

import pandas as pd


DEMO_REASONS = (
    "Price",
    "Service quality",
    "Support response",
    "Missing features",
    "Competitor offer",
    "Moving or eligibility",
    "Other",
)
DEMO_ACTIONS = (
    "Discount",
    "Plan change",
    "Technical fix",
    "Support callback",
    "Feature guidance",
    "Other",
    "None",
)


def build_demo_customers(seed: int = 2026, size: int = 480) -> pd.DataFrame:
    rng = random.Random(seed)
    plans = {
        "Basic": (39.0, 0.37),
        "Plus": (64.0, 0.43),
        "Premium": (92.0, 0.20),
    }
    contracts = ("Month-to-month", "One year", "Two year")
    contract_weights = (0.53, 0.30, 0.17)
    rows = []

    for index in range(1, size + 1):
        plan = rng.choices(list(plans), weights=[value[1] for value in plans.values()])[0]
        base_charge = plans[plan][0]
        contract = rng.choices(contracts, weights=contract_weights)[0]
        tenure = rng.randint(1, 72)
        support_contacts = min(9, int(rng.expovariate(1 / 1.8)))
        escalations = min(support_contacts, rng.choices([0, 1, 2, 3], [0.58, 0.27, 0.11, 0.04])[0])
        satisfaction = max(1.0, min(5.0, round(rng.gauss(3.55, 0.9), 1)))
        engagement = max(5, min(100, round(rng.gauss(64, 18))))
        monthly_charge = round(max(24.0, rng.gauss(base_charge, base_charge * 0.12)), 2)

        log_odds = (
            -2.05
            + max(0, 24 - tenure) * 0.025
            + (0.78 if contract == "Month-to-month" else 0)
            + max(0, 3.5 - satisfaction) * 0.48
            + max(0, 50 - engagement) * 0.012
            + escalations * 0.28
            + support_contacts * 0.035
        )
        churn_probability = 1 / (1 + exp(-log_odds))

        rows.append(
            {
                "customer_id": f"DEMO-CUST-{index:04d}",
                "tenure": tenure,
                "contract_type": contract,
                "plan": plan,
                "monthly_charge": monthly_charge,
                "support_contacts": support_contacts,
                "escalations": escalations,
                "satisfaction_score": satisfaction,
                "engagement_score": engagement,
                "churn_target": int(rng.random() < churn_probability),
            }
        )

    return pd.DataFrame(rows)


def build_demo_cases(
    customers: pd.DataFrame,
    as_of: date | None = None,
    seed: int = 72026,
) -> pd.DataFrame:
    if customers.empty:
        raise ValueError("Demo cases need at least one synthetic customer.")

    rng = random.Random(seed)
    today = as_of or date.today()
    outcomes = (
        "Saved", "Still leaving", "Pending", "Saved", "Still leaving",
        "Saved", "Pending", "Still leaving", "Saved", "Still leaving",
        "Pending", "Saved", "Still leaving", "Saved", "Pending",
        "Still leaving", "Saved", "Still leaving", "Pending", "Saved",
        "Still leaving",
    )
    cancellations = (
        "Not cancelled", "Cancelled", "Not cancelled", "Not cancelled",
        "Cancelled", "Not cancelled", "Not cancelled", "Cancelled",
        "Not cancelled", "Cancelled", "Not cancelled", "Not cancelled",
        "Cancelled", "Not cancelled", "Not cancelled", "Cancelled",
        "Not cancelled", "Cancelled", "Not cancelled", "Not cancelled",
        "Cancelled",
    )
    customer_records = customers.to_dict(orient="records")
    rows = []

    for index, (outcome, cancellation_status) in enumerate(
        zip(outcomes, cancellations), start=1
    ):
        customer = customer_records[(index * 19) % len(customer_records)]
        days_ago = rng.randint(12, 360)
        if cancellation_status == "Cancelled":
            days_ago = max(days_ago, rng.randint(35, 110))
        request_date = today - timedelta(days=days_ago)
        cancellation_date = None
        if cancellation_status == "Cancelled":
            cancellation_date = min(
                today,
                request_date + timedelta(days=rng.randint(3, 24)),
            )
        reason = DEMO_REASONS[(index - 1) % len(DEMO_REASONS)]
        action = DEMO_ACTIONS[(index * 3) % len(DEMO_ACTIONS)]
        rows.append(
            {
                "case_id": index,
                "customer_id": customer["customer_id"],
                "request_date": request_date.isoformat(),
                "plan": customer["plan"],
                "contract_type": customer["contract_type"],
                "tenure": customer["tenure"],
                "monthly_charge": customer["monthly_charge"],
                "primary_reason": reason,
                "secondary_reason": None,
                "customer_comments": "Synthetic example for interface demonstration.",
                "stay_request": None,
                "retention_action": action,
                "outcome": outcome,
                "cancellation_status": cancellation_status,
                "cancellation_date": (
                    cancellation_date.isoformat() if cancellation_date else None
                ),
                "owner": "Demo team",
                "follow_up_date": (
                    (request_date + timedelta(days=7)).isoformat()
                    if outcome == "Pending"
                    else None
                ),
                "notes": None,
                "created_at": request_date.isoformat() + "T09:00:00Z",
                "updated_at": request_date.isoformat() + "T09:00:00Z",
                "data_classification": "DEMO DATA — NOT REAL CUSTOMERS",
            }
        )

    return pd.DataFrame(rows)
