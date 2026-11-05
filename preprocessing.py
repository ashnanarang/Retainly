from typing import Literal, overload

import pandas as pd


def clean_data(df):
    df = df.copy()
    df.columns = pd.Index([str(col).strip() for col in df.columns])
    if not df.columns.is_unique:
        raise ValueError("The CSV has column names that become duplicates after trimming whitespace.")
    for col in df.select_dtypes(
        include="object"
    ).columns:
        df[col] = df[col].apply(
            lambda x: x.strip()
            if isinstance(x, str)
            else x
        )

    return df


@overload
def convert_churn_target(
    df: pd.DataFrame,
    churn_column: str,
    return_unmapped: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def convert_churn_target(
    df: pd.DataFrame,
    churn_column: str,
    return_unmapped: Literal[True],
) -> tuple[pd.DataFrame, dict[str, int]]: ...


def convert_churn_target(
    df,
    churn_column,
    return_unmapped=False,
):
    df = df.copy()
    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "churned": 1,
        "churn": 1,
        "cancelled": 1,
        "canceled": 1,
        "left": 1,
        "exited": 1,
        "active": 0,
        "retained": 0,
        "stayed": 0,
        "not churned": 0,
        "not_churned": 0,
        "1": 1,
        "0": 0
    }

    def map_value(value):
        if pd.isna(value):
            return pd.NA
        normalized = str(value).strip().lower()
        if normalized in mapping:
            return mapping[normalized]
        try:
            numeric = float(normalized)
        except ValueError:
            return pd.NA
        return int(numeric) if numeric in (0, 1) else pd.NA

    converted = df[churn_column].map(map_value).astype("Int64")
    unmapped = (
        df.loc[df[churn_column].notna() & converted.isna(), churn_column]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    df["churn_target"] = converted
    if return_unmapped:
        return df, unmapped
    return df