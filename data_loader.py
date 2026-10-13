"""
Data Loader Module for Online Shoppers Purchasing Intention Dataset
Handles loading, initial verification, data dictionary generation, and integrity checks.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


DATA_DICTIONARY_INFO = [
    {"Column": "Administrative", "Type": "Numeric (int)", "Description": "Number of administrative pages visited", "Role": "Feature"},
    {"Column": "Administrative_Duration", "Type": "Numeric (float)", "Description": "Total duration spent on administrative pages (seconds)", "Role": "Feature"},
    {"Column": "Informational", "Type": "Numeric (int)", "Description": "Number of informational pages visited", "Role": "Feature"},
    {"Column": "Informational_Duration", "Type": "Numeric (float)", "Description": "Total duration spent on informational pages (seconds)", "Role": "Feature"},
    {"Column": "ProductRelated", "Type": "Numeric (int)", "Description": "Number of product-related catalog pages visited", "Role": "Feature"},
    {"Column": "ProductRelated_Duration", "Type": "Numeric (float)", "Description": "Total duration spent browsing product pages (seconds)", "Role": "Feature"},
    {"Column": "BounceRates", "Type": "Numeric (float)", "Description": "Average bounce rate percentage of visited pages", "Role": "Feature"},
    {"Column": "ExitRates", "Type": "Numeric (float)", "Description": "Average exit rate percentage of visited pages", "Role": "Feature"},
    {"Column": "PageValues", "Type": "Numeric (float)", "Description": "Average Google Analytics page value of visited pages", "Role": "Feature"},
    {"Column": "SpecialDay", "Type": "Numeric (float)", "Description": "Closeness of browsing site visit to a festival/holiday (0 to 1)", "Role": "Feature"},
    {"Column": "Month", "Type": "Categorical", "Description": "Calendar month of the browsing session (Feb - Dec)", "Role": "Feature"},
    {"Column": "OperatingSystems", "Type": "Discrete (int)", "Description": "Operating system identifier of visitor (1 to 8)", "Role": "Feature"},
    {"Column": "Browser", "Type": "Discrete (int)", "Description": "Web browser identifier of visitor (1 to 13)", "Role": "Feature"},
    {"Column": "Region", "Type": "Discrete (int)", "Description": "Geographic region code of visitor (1 to 9)", "Role": "Feature"},
    {"Column": "TrafficType", "Type": "Discrete (int)", "Description": "Web traffic acquisition source type (1 to 20)", "Role": "Feature"},
    {"Column": "VisitorType", "Type": "Categorical", "Description": "Visitor classification (Returning_Visitor, New_Visitor, Other)", "Role": "Feature"},
    {"Column": "Weekend", "Type": "Boolean", "Description": "Boolean flag indicating if session occurred on weekend", "Role": "Feature"},
    {"Column": "Revenue", "Type": "Boolean Target", "Description": "Target: Did the session culminate in a purchase/transaction?", "Role": "Target"}
]


def load_raw_data(filepath: str = "data/online_shoppers_intention.csv") -> pd.DataFrame:
    """
    Loads the Online Shoppers Purchasing Intention dataset from a local CSV file.
    Validates file presence and verifies expected column schema.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please ensure data has been downloaded into the data/ directory."
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading CSV file at '{filepath}': {e}")

    expected_cols = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
        "VisitorType", "Weekend", "Revenue"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    return df


def get_data_dictionary() -> pd.DataFrame:
    """Returns the comprehensive data dictionary as a pandas DataFrame."""
    return pd.DataFrame(DATA_DICTIONARY_INFO)


def verify_data_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive integrity checks on the dataframe:
    - Dimensions (rows, columns)
    - Column types (numeric vs categorical/discrete)
    - Missing / null counts and percentages
    - Duplicates count
    - Target distribution (Revenue) and class imbalance ratio
    """
    num_rows, num_cols = df.shape
    num_cols_list = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_list = df.select_dtypes(include=["object", "category", "bool", "string", str]).columns.tolist()

    # Null counts
    null_counts = df.isnull().sum().to_dict()
    null_pcts = (df.isnull().mean() * 100).round(3).to_dict()

    # Duplicates
    dup_count = int(df.duplicated().sum())

    # Target stats
    target_dist = {}
    imbalance_ratio = 1.0
    if "Revenue" in df.columns:
        counts = df["Revenue"].value_counts()
        target_dist = {str(k): int(v) for k, v in counts.items()}
        if len(counts) == 2 and counts.min() > 0:
            imbalance_ratio = round(float(counts.max() / counts.min()), 2)

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "numeric_features": num_cols_list,
        "categorical_features": cat_cols_list,
        "null_counts": null_counts,
        "null_percentages": null_pcts,
        "duplicate_count": dup_count,
        "target_distribution": target_dist,
        "class_imbalance_ratio": imbalance_ratio
    }


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns an overview summary of the dataset suitable for reports and dashboards.
    """
    integrity = verify_data_integrity(df)
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    return {
        "dataset_name": "UCI Online Shoppers Purchasing Intention Dataset",
        "source": "UC Irvine Machine Learning Repository (DOI: 10.24432/C5F88Q)",
        "rows": integrity["num_rows"],
        "columns": integrity["num_cols"],
        "numeric_count": len(integrity["numeric_features"]),
        "categorical_count": len(integrity["categorical_features"]),
        "memory_usage_mb": memory_mb,
        "duplicates": integrity["duplicate_count"],
        "class_imbalance_ratio": integrity["class_imbalance_ratio"],
        "target_distribution": integrity["target_distribution"]
    }
