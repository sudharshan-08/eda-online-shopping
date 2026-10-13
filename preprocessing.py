"""
Preprocessing Pipeline Module for Online Shoppers Purchasing Intention Dataset
Handles deduplication, outlier detection & Winsorization capping, categorical encoding,
leakage-free stratified train/val/test splitting, and robust feature scaling.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Detects and removes identical duplicate browsing session records.
    Returns the cleaned DataFrame and the number of duplicates removed.
    """
    initial_count = len(df)
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_count - len(df_cleaned)
    return df_cleaned, duplicates_removed


def detect_outliers_iqr(df: pd.DataFrame, num_cols: List[str], factor: float = 1.5) -> Dict[str, Dict[str, Any]]:
    """
    Detects outliers across numerical e-commerce session features using Tukey's IQR method.
    Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1.
    Outliers lie below (Q1 - 1.5*IQR) or above (Q3 + 1.5*IQR).
    """
    results = {}
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)

        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        count = int(outliers_mask.sum())
        percentage = round((count / len(df)) * 100, 2)

        results[col] = {
            "q1": round(float(q1), 3),
            "q3": round(float(q3), 3),
            "iqr": round(float(iqr), 3),
            "lower_bound": round(float(lower_bound), 3),
            "upper_bound": round(float(upper_bound), 3),
            "outlier_count": count,
            "outlier_pct": percentage
        }
    return results


def detect_outliers_zscore(df: pd.DataFrame, num_cols: List[str], threshold: float = 3.0) -> Dict[str, Dict[str, Any]]:
    """
    Detects outliers using standard Z-score (|z| > 3.0).
    """
    results = {}
    for col in num_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            results[col] = {"outlier_count": 0, "outlier_pct": 0.0}
            continue
        z_scores = np.abs((df[col] - mean) / std)
        count = int((z_scores > threshold).sum())
        pct = round((count / len(df)) * 100, 2)
        results[col] = {
            "mean": round(float(mean), 3),
            "std": round(float(std), 3),
            "outlier_count": count,
            "outlier_pct": pct
        }
    return results


def treat_outliers_capping(
    df: pd.DataFrame,
    num_cols: List[str],
    factor: float = 1.5,
    exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Treats outliers using IQR capping (Winsorization).
    Values below Q1 - 1.5*IQR are capped at lower bound;
    values above Q3 + 1.5*IQR are capped at upper bound.
    
    Rationale:
    In e-commerce web browsing, users who spend significant time (ProductRelated_Duration)
    or visit high-value pages (PageValues) are the highest-converting buyers.
    Discarding these rows would strip away rare positive purchasers. Capping mitigates
    extreme statistical variance while retaining all genuine visitor sessions.
    """
    if exclude_cols is None:
        exclude_cols = []

    df_treated = df.copy()
    for col in num_cols:
        if col in exclude_cols:
            continue
        q1 = df_treated[col].quantile(0.25)
        q3 = df_treated[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(0.0, q1 - (factor * iqr))  # Page counts & durations are non-negative
        upper_bound = q3 + (factor * iqr)

        df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
    return df_treated


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Encodes categorical and discrete features and extracts the binary target 'Revenue'.
    - Categorical features (Month, VisitorType, Weekend, OperatingSystems, Browser, Region, TrafficType)
      are converted via One-Hot Encoding.
    - Boolean target 'Revenue' is converted to integer (True -> 1, False -> 0).
    """
    df_proc = df.copy()

    # Separate target
    if "Revenue" in df_proc.columns:
        y = df_proc["Revenue"].astype(int)
        X = df_proc.drop(columns=["Revenue"])
    else:
        y = None
        X = df_proc

    # Convert Weekend boolean to integer
    if "Weekend" in X.columns:
        X["Weekend"] = X["Weekend"].astype(int)

    # Columns to treat as categorical for one-hot encoding
    cat_cols = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
    existing_cat = [c for c in cat_cols if c in X.columns]
    
    # Cast discrete IDs to string so get_dummies expands them properly
    for c in existing_cat:
        X[c] = X[c].astype(str)

    X_encoded = pd.get_dummies(X, columns=existing_cat, drop_first=False, dtype=float)

    return X_encoded, y


def split_and_scale_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    scaler_type: str = "robust"
) -> Dict[str, Any]:
    """
    Splits data into stratified train (70%), validation (15%), and test (15%) subsets.
    Fits feature scaler STRICTLY on the training set (X_train) and transforms
    val and test sets to guarantee ZERO data leakage.
    """
    # Step 1: Split into Train + Val (85%) and Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Step 2: Split Train + Val into Train (70% total) and Val (15% total)
    val_rel_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_rel_size,
        stratify=y_train_val,
        random_state=random_state
    )

    # Step 3: Fit scaler strictly on training set
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    feature_names = X.columns.tolist()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "y_train": y_train.values,
        "X_val": X_val_scaled,
        "y_val": y_val.values,
        "X_test": X_test_scaled,
        "y_test": y_test.values,
        "X_train_df": X_train,
        "X_val_df": X_val,
        "X_test_df": X_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "class_counts_train": {
            0: int((y_train == 0).sum()),
            1: int((y_train == 1).sum())
        }
    }


def run_full_preprocessing_pipeline(
    raw_path: str = "data/online_shoppers_intention.csv",
    save_cleaned_path: str = "data/shoppers_cleaned.csv"
) -> Dict[str, Any]:
    """
    Executes the complete preprocessing pipeline end-to-end:
    1. Loads raw dataset.
    2. Removes duplicate session records.
    3. Detects outliers.
    4. Treats outliers via Winsorization capping.
    5. Saves cleaned dataset to disk.
    6. Encodes categorical variables and performs leakage-free stratified split & scaling.
    """
    from src.data_loader import load_raw_data

    df_raw = load_raw_data(raw_path)
    df_dedup, dups_removed = remove_duplicates(df_raw)

    continuous_cols = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    outlier_stats = detect_outliers_iqr(df_dedup, continuous_cols)

    # Treat outliers via capping
    df_cleaned = treat_outliers_capping(df_dedup, continuous_cols)

    # Save cleaned dataset for reproducible analysis & dashboard
    os.makedirs(os.path.dirname(save_cleaned_path), exist_ok=True)
    df_cleaned.to_csv(save_cleaned_path, index=False)

    # One-hot encoding & feature preparation
    X_encoded, y = prepare_features(df_cleaned)

    # Stratified split and scaling
    split_artifacts = split_and_scale_data(X_encoded, y, test_size=0.15, val_size=0.15, random_state=42)
    split_artifacts["duplicates_removed"] = dups_removed
    split_artifacts["outlier_stats"] = outlier_stats
    split_artifacts["df_cleaned"] = df_cleaned

    return split_artifacts
