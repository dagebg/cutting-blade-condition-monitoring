"""
Model definition and training utilities for the cutting blade condition
monitoring project.

This module defines the machine learning models used for operating-mode
classification, anomaly detection, and degradation estimation.

Responsibilities include:
- Constructing model instances (e.g. tree-based models, linear models, baselines).
- Providing training and prediction helpers for tabular feature data.
- Handling model hyperparameters and simple search routines.
- Saving and loading trained model artifacts when needed.

By collecting model-related code here, notebooks can focus on experiment
orchestration while this module encapsulates the modeling details.
"""

from typing import List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline

import pandas as pd

# ---------------------------------------------------------------------------
# Kept for EDA / display purposes only — no longer used for model training
# ---------------------------------------------------------------------------
def standardize_features_with_renaming(
    feature_df: pd.DataFrame, label_cols: List[str]
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize numerical columns, keep labels, and rename all columns.

    This mirrors the notebook logic exactly:
      - All columns not in `label_cols` are standardized.
      - Label columns are copied back unchanged.
      - The final DataFrame's columns are overwritten with `new_column_names`
        in the existing column order (numerical columns first, then labels).

    Args:
        feature_df: DataFrame containing numerical feature columns and labels.
        label_cols: Column names that should not be standardized.

    Returns:
        A tuple of:
            - The standardized and renamed DataFrame.
            - The fitted StandardScaler instance.
    """
    # exclude columns for which standardization does not make sense
    numerical_cols = [col for col in feature_df.columns if col not in label_cols]

    # extracting feature matrix
    X = feature_df[numerical_cols].values

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # combine scaled numerical features with original label columns
    feature_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols)
    for label in label_cols:
        feature_scaled_df[label] = feature_df[label].values

    # change column names for better readability
    feature_scaled_df.columns = [
        "K_PF_mean",
        "K_PF_std",
        "K_PF_min",
        "K_PF_max",
        "K_PF_outl",
        "K_SPD_mean",
        "K_SPD_std",
        "K_SPD_min",
        "K_SPD_max",
        "K_SPD_outl",
        "K_POS_mean",
        "K_POS_std",
        "K_POS_min",
        "K_POS_max",
        "K_POS_outl",
        "K_TOR_mean",
        "K_TOR_std",
        "K_TOR_min",
        "K_TOR_max",
        "K_TOR_outl",
        "F_PF_mean",
        "F_PF_std",
        "F_PF_min",
        "F_PF_max",
        "F_PF_outl",
        "F_SPD_mean",
        "F_SPD_std",
        "F_SPD_min",
        "F_SPD_max",
        "F_SPD_outl",
        "F_POS_mean",
        "F_POS_std",
        "F_POS_min",
        "F_POS_max",
        "F_POS_outl",
        "mode",
        "degradation",
        "file_anomaly",
    ]

    return feature_scaled_df, scaler


# ---------------------------------------------------------------------------
# 4.1  Classification — Random Forest
# ---------------------------------------------------------------------------
def run_random_forest_mode_classification(X: pd.DataFrame, y: pd.Series, cv):
    """
    Trains a Pipeline(StandardScaler → RandomForestClassifier) per CV fold.
    The scaler is fitted on training data only — no leakage.

    Returns a list of dicts: [{fold, y_val, y_pred}, ...]
    """
    print("Model: Pipeline(StandardScaler → RandomForestClassifier)")
    print("Parameters: n_estimators=100, random_state=42\n")

    predictions = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )

        print(f"Fold {fold + 1}")
        print(
            f"  Training on {len(train_idx)} samples, Validation on {len(val_idx)} samples"
        )
        pipeline.fit(X_train, y_train)
        print("  model training done")
        y_pred = pipeline.predict(X_val)
        print("  predictions generated\n")

        predictions.append(
            {
                "fold": fold + 1,
                "y_val": y_val.reset_index(drop=True),
                "y_pred": y_pred,
            }
        )

    print(f"Finished modeling — {len(predictions)} trained folds")
    return predictions


# ---------------------------------------------------------------------------
# 4.2  Regression — Ridge Degradation
# ---------------------------------------------------------------------------
def run_ridge_degradation_regression(
    X: pd.DataFrame, y: pd.Series, y_cat: pd.Series, cv
):
    """
    Trains a Pipeline(StandardScaler → Ridge) per CV fold.
    y_cat is used only for stratified splitting; y is the regression target.
    The scaler is fitted on training data only — no leakage.

    Returns a list of dicts: [{fold, y_val, y_pred}, ...]
    """
    print("model initialised: Pipeline(StandardScaler → Ridge Regression)")
    print("parameter standard: alpha=1.0\n")

    predictions = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_cat)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0)),
            ]
        )

        print(f"Fold {fold + 1}")
        print(
            f"  Training on {len(train_idx)} Samples, Validation on {len(val_idx)} Samples"
        )
        pipeline.fit(X_train, y_train)
        print("  model training done")
        y_pred = pipeline.predict(X_val)
        print("  predictions generated\n")

        predictions.append(
            {
                "fold": fold + 1,
                "y_val": y_val.reset_index(drop=True),
                "y_pred": y_pred,
            }
        )

    print(f"Finished modeling — {len(predictions)} trained folds")
    return predictions


# ---------------------------------------------------------------------------
# 4.3  Anomaly Detection — Isolation Forest
# ---------------------------------------------------------------------------
def run_isolation_forest_anomaly_detection(X: pd.DataFrame, y: pd.Series, cv):
    """
    Trains a Pipeline(StandardScaler → IsolationForest) per CV fold.
    The model is unsupervised; y is used only for stratified CV splitting.
    IsolationForest.predict() returns 1 (normal) / -1 (anomaly) → converted to 0/1.
    The scaler is fitted on training data only — no leakage.

    Returns a list of dicts: [{fold, y_val, y_pred}, ...]
    """
    print("model initialised: Pipeline(StandardScaler → Isolation Forest)")
    print("parameter: contamination=auto, random_state=42\n")

    predictions = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("iso", IsolationForest(contamination="auto", random_state=42)),
            ]
        )

        print(f"Fold {fold + 1}")
        print(
            f"  Training on {len(train_idx)} Samples, Validation auf {len(val_idx)} Samples"
        )
        # IsolationForest is unsupervised — fit on X_train only
        pipeline.fit(X_train)
        print("  model training done")
        # Convert sklearn convention: -1 → 1 (anomaly), 1 → 0 (normal)
        raw_pred = pipeline.predict(X_val)
        y_pred = (raw_pred == -1).astype(int)
        print("  predictions generated\n")

        predictions.append(
            {
                "fold": fold + 1,
                "y_val": y_val.reset_index(drop=True),
                "y_pred": y_pred,
            }
        )

    print(f"Finished modeling — {len(predictions)} trained folds")
    return predictions
