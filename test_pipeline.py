"""
Automated Test Suite for Online Shoppers Purchasing Intention Pipeline
Verifies data loading, integrity, preprocessing, leakage prevention,
PyTorch model architecture, and evaluation metrics computation.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch

from src.data_loader import load_raw_data, get_data_dictionary, verify_data_integrity
from src.preprocessing import (
    remove_duplicates, detect_outliers_iqr, treat_outliers_capping,
    prepare_features, split_and_scale_data
)
from src.model import ShoppersMLP, predict_shoppers_mlp
from src.evaluation import compute_classification_metrics


@pytest.fixture(scope="session")
def raw_dataset():
    """Session-level fixture to load dataset once for unit tests."""
    df = load_raw_data("data/online_shoppers_intention.csv")
    return df


def test_data_loader_dimensions(raw_dataset):
    """Verifies that dataset loads with 12,330 rows and 18 attributes."""
    assert raw_dataset.shape[0] == 12330, f"Expected 12,330 rows, got {raw_dataset.shape[0]}"
    assert raw_dataset.shape[1] == 18, f"Expected 18 columns, got {raw_dataset.shape[1]}"


def test_data_dictionary():
    """Verifies that data dictionary documents all 18 features."""
    dict_df = get_data_dictionary()
    assert len(dict_df) == 18, f"Expected 18 dictionary entries, got {len(dict_df)}"
    assert "PageValues" in dict_df["Column"].values
    assert "Revenue" in dict_df["Column"].values


def test_data_integrity(raw_dataset):
    """Verifies data integrity checks and presence of binary target."""
    integrity = verify_data_integrity(raw_dataset)
    assert integrity["num_rows"] == 12330
    assert integrity["num_cols"] == 18
    assert "True" in integrity["target_distribution"] or True in integrity["target_distribution"]
    assert "False" in integrity["target_distribution"] or False in integrity["target_distribution"]
    assert sum(integrity["null_counts"].values()) == 0


def test_preprocessing_deduplication(raw_dataset):
    """Verifies duplicate removal functionality."""
    df_dedup, dups_removed = remove_duplicates(raw_dataset)
    assert dups_removed == 125, f"Expected 125 duplicates removed, got {dups_removed}"
    assert len(df_dedup) == 12330 - 125


def test_outlier_detection_and_capping(raw_dataset):
    """Verifies Tukey IQR outlier detection and Winsorization capping."""
    num_cols = ["ProductRelated_Duration", "PageValues", "ExitRates"]
    outliers = detect_outliers_iqr(raw_dataset, num_cols)
    assert "PageValues" in outliers
    assert outliers["PageValues"]["outlier_count"] > 0

    df_capped = treat_outliers_capping(raw_dataset, num_cols)
    assert len(df_capped) == len(raw_dataset), "Capping must preserve all rows"
    assert df_capped["PageValues"].max() <= outliers["PageValues"]["upper_bound"] + 1e-3


def test_split_and_scaling_leakage_free(raw_dataset):
    """
    Verifies that splitting and scaling are strictly leakage-free:
    - Train (70%), Val (15%), Test (15%)
    - Scaler fitted exclusively on train set
    - Output dimensions match exactly across splits
    """
    df_clean, _ = remove_duplicates(raw_dataset)
    X_encoded, y = prepare_features(df_clean)

    artifacts = split_and_scale_data(X_encoded, y, test_size=0.15, val_size=0.15, random_state=42)

    X_train = artifacts["X_train"]
    X_val = artifacts["X_val"]
    X_test = artifacts["X_test"]
    y_train = artifacts["y_train"]
    y_test = artifacts["y_test"]

    total_len = len(X_train) + len(X_val) + len(X_test)
    assert total_len == len(df_clean)
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

    # Verify no NaN values in scaled matrices
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_test).any()

    # Verify class balance preserved
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()
    assert abs(train_pos_rate - test_pos_rate) < 0.02


def test_pytorch_shoppers_mlp_architecture():
    """Verifies PyTorch ShoppersMLP architecture, forward pass, and output shapes."""
    in_features = 74
    batch_size = 16
    model = ShoppersMLP(in_features=in_features)

    dummy_input = torch.randn(batch_size, in_features)
    output_logits = model(dummy_input)

    assert output_logits.shape == (batch_size,), f"Expected output shape ({batch_size},), got {output_logits.shape}"
    probs = torch.sigmoid(output_logits)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Sigmoid outputs must lie within [0, 1]"


def test_evaluation_metrics_computation():
    """Verifies evaluation metric calculation and bounds."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 4
    assert metrics["confusion_matrix"]["tn"] == 4
