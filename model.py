import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEAKAGE_TERMS = (
    "churn",
    "cancel",
    "termination",
    "retention_action",
    "post_churn",
    "exit_date",
    "end_date",
)
IDENTIFIER_NAMES = {
    "id",
    "customerid",
    "accountid",
    "subscriberid",
    "userid",
    "memberid",
    "clientid",
    "recordid",
    "rowid",
    "policyid",
}


def eligible_feature_columns(df, target="churn_target"):
    features = []
    excluded = {}
    for column in df.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
        if column == target:
            excluded[column] = "target"
        elif normalized == "id" or normalized.endswith("_id") or normalized in IDENTIFIER_NAMES:
            excluded[column] = "identifier"
        elif any(term in normalized for term in LEAKAGE_TERMS):
            excluded[column] = "possible post-outcome leakage"
        else:
            features.append(column)
    return features, excluded


def _build_pipeline(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def _metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "confusion_matrix": matrix,
        "threshold": threshold,
    }


def evaluate_threshold(y_true, probabilities, threshold):
    return _metrics(y_true, probabilities, threshold)


def train_model(df, target="churn_target", threshold=0.5):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    if not 0 < threshold < 1:
        raise ValueError("Decision threshold must be between 0 and 1.")

    data = df.loc[df[target].notna()].copy()
    y = pd.to_numeric(data[target], errors="coerce")
    valid = y.isin([0, 1])
    data = data.loc[valid]
    y = y.loc[valid].astype(int)
    if y.nunique() != 2:
        raise ValueError("Model training requires both target classes (0 and 1).")
    min_class_count = int(y.value_counts().min())
    if min_class_count < 2:
        raise ValueError("At least two examples of each target class are needed for stratified validation.")

    features, excluded = eligible_feature_columns(data, target)
    if not features:
        raise ValueError("No eligible model features remain after excluding IDs and leakage fields.")
    X = data[features].copy()
    folds = min(5, min_class_count)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    model = _build_pipeline(X)
    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    metrics = _metrics(y.to_numpy(), probabilities, threshold)
    metrics.update(
        {
            "cv_folds": folds,
            "validation": "out-of-fold stratified cross-validation",
            "sample_count": len(y),
            "feature_columns": features,
            "excluded_columns": excluded,
            "validation_targets": y.to_numpy(),
            "validation_probabilities": probabilities,
        }
    )
    model.fit(X, y)
    return model, metrics


def score_customers(model, customers, feature_columns):
    missing_columns = [column for column in feature_columns if column not in customers.columns]
    if missing_columns:
        raise ValueError(
            "Scoring data is missing required model features: "
            + ", ".join(map(str, missing_columns))
        )
    X = customers.loc[:, feature_columns]
    return model.predict_proba(X)[:, 1]
