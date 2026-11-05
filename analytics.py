import pandas as pd


def churn_rate(df):

    if "churn_target" not in df.columns:
        return None

    return (
        df["churn_target"].mean() * 100
    )


def revenue_at_risk(
    df,
    churn_column,
    charge_column
):

    if (
        churn_column not in df.columns
        or charge_column not in df.columns
    ):
        return None

    churned = df[
        df[churn_column] == 1
    ]

    return churned[
        charge_column
    ].sum()


def churn_by_category(
    df,
    category,
    churn_column
):

    if (
        category not in df.columns
        or churn_column not in df.columns
    ):
        return None

    return pd.crosstab(
        df[category],
        df[churn_column]
    )


def average_by_churn(
    df,
    numeric_column
):

    if (
        numeric_column not in df.columns
        or "churn_target" not in df.columns
    ):
        return None

    return (
        df.groupby("churn_target")
        [numeric_column]
        .mean()
    )