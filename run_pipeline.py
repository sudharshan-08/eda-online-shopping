"""
Master Pipeline Runner Script for Online Shoppers Purchasing Intention Dataset
Executes data preprocessing, EDA, statistical hypothesis testing,
PyTorch ShoppersMLP training, baseline benchmarking, figure generation, and metric reporting.
"""

import os
import json
import numpy as np
import pandas as pd
import torch

from src.data_loader import load_raw_data, get_dataset_summary, get_data_dictionary, verify_data_integrity
from src.preprocessing import run_full_preprocessing_pipeline
from src.eda import (
    compute_univariate_stats, compute_bivariate_numeric,
    compute_bivariate_categorical, compute_correlation_matrices,
    check_multicollinearity_vif, test_probability_distributions
)
from src.visualizations import generate_all_eda_visualizations
from src.model import train_shoppers_mlp, predict_shoppers_mlp, train_baseline_models
from src.evaluation import (
    compute_classification_metrics, plot_loss_curves,
    plot_accuracy_curves, plot_confusion_matrix_heatmap,
    plot_roc_curve_viz, plot_precision_recall_viz,
    plot_model_comparison_benchmark
)


def run_pipeline():
    print("=" * 70)
    print("STARTING COMPLETE EDA & DEEP LEARNING MODEL PIPELINE")
    print("Case Study: Online Shoppers Purchasing Intention Dataset | KPRIET")
    print("=" * 70)

    # 1. Load Data & Summary
    print("\n[Step 1/7] Loading Dataset & Verifying Data Integrity...")
    df_raw = load_raw_data("data/online_shoppers_intention.csv")
    summary = get_dataset_summary(df_raw)
    integrity = verify_data_integrity(df_raw)
    print(f"Loaded {summary['rows']:,} rows and {summary['columns']} features.")
    print(f"Target distribution (Revenue): {summary['target_distribution']} (Imbalance Ratio: {summary['class_imbalance_ratio']}:1)")

    # 2. Preprocessing & Leakage-Free Splitting
    print("\n[Step 2/7] Preprocessing, Outlier Treatment & Leakage-Free Splitting...")
    artifacts = run_full_preprocessing_pipeline(
        raw_path="data/online_shoppers_intention.csv",
        save_cleaned_path="data/shoppers_cleaned.csv"
    )
    df_cleaned = artifacts["df_cleaned"]
    X_train, y_train = artifacts["X_train"], artifacts["y_train"]
    X_val, y_val = artifacts["X_val"], artifacts["y_val"]
    X_test, y_test = artifacts["X_test"], artifacts["y_test"]
    print(f"Duplicates removed: {artifacts['duplicates_removed']}")
    print(f"Splits: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    print(f"Engineered input feature dimensions: {artifacts['n_features']}")

    # 3. Exploratory Data Analysis & Statistical Testing
    print("\n[Step 3/7] Computing EDA Statistics & Hypothesis Tests...")
    univ_stats = compute_univariate_stats(df_cleaned)
    biv_numeric = compute_bivariate_numeric(df_cleaned)
    biv_cat = compute_bivariate_categorical(df_cleaned)
    pearson_corr, spearman_corr = compute_correlation_matrices(df_cleaned)
    vif_df = check_multicollinearity_vif(df_cleaned)
    dist_fit_pagevalues = test_probability_distributions(df_cleaned, "PageValues")
    dist_fit_product_dur = test_probability_distributions(df_cleaned, "ProductRelated_Duration")

    eda_summary = {
        "dataset_summary": summary,
        "univariate_stats": univ_stats.to_dict(orient="records"),
        "bivariate_numeric": biv_numeric.to_dict(orient="records"),
        "bivariate_categorical": biv_cat,
        "vif_analysis": vif_df.to_dict(orient="records"),
        "distribution_fit_pagevalues": dist_fit_pagevalues,
        "distribution_fit_product_dur": dist_fit_product_dur
    }

    os.makedirs("outputs/reports", exist_ok=True)
    with open("outputs/reports/eda_summary.json", "w") as f:
        json.dump(eda_summary, f, indent=2)
    print("EDA statistical summary saved to outputs/reports/eda_summary.json")

    # 4. Generate EDA Visualizations
    print("\n[Step 4/7] Generating High-Resolution EDA Visualizations (300 DPI)...")
    generate_all_eda_visualizations(df_cleaned, output_dir="outputs/figures")
    print("Saved EDA figures to outputs/figures/ (7 figures generated).")

    # 5. Train Deep Learning MLP Model
    print("\n[Step 5/7] Training PyTorch ShoppersMLP with Early Stopping & AdamW...")
    model, history = train_shoppers_mlp(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=128,
        epochs=35,
        lr=0.001,
        weight_decay=1e-4,
        patience=6,
        seed=42,
        save_model_path="outputs/reports/best_mlp_model.pt"
    )
    epochs_trained = len(history["epoch"])
    best_val_loss = min(history["val_loss"])
    print(f"ShoppersMLP Training concluded at epoch {epochs_trained}. Best validation loss: {best_val_loss:.4f}")

    # 6. Train Baseline Models
    print("\n[Step 6/7] Training Baseline Models (Logistic Regression & Random Forest)...")
    baselines = train_baseline_models(X_train, y_train, seed=42)

    # 7. Evaluate on Held-Out Test Set
    print("\n[Step 7/7] Evaluating All Models on Held-Out Test Set...")
    # Evaluate ShoppersMLP
    mlp_test_probs = predict_shoppers_mlp(model, X_test)
    mlp_metrics = compute_classification_metrics(y_test, mlp_test_probs, threshold=0.5)

    # Evaluate Logistic Regression
    lr_model = baselines["logistic_regression"]
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_metrics = compute_classification_metrics(y_test, lr_probs, threshold=0.5)

    # Evaluate Random Forest
    rf_model = baselines["random_forest"]
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = compute_classification_metrics(y_test, rf_probs, threshold=0.5)

    benchmark_dict = {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "PyTorch ShoppersMLP": mlp_metrics
    }

    print("\n" + "=" * 50)
    print("TEST SET BENCHMARK RESULTS SUMMARY:")
    print("=" * 50)
    for model_name, m in benchmark_dict.items():
        print(f"Model: {model_name:22} | Acc: {m['accuracy']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | ROC-AUC: {m['roc_auc']:.4f}")

    # Save Metrics JSON
    final_report = {
        "benchmark": benchmark_dict,
        "mlp_training_history": history,
        "test_size": len(X_test),
        "class_counts_test": {
            0: int((y_test == 0).sum()),
            1: int((y_test == 1).sum())
        }
    }
    with open("outputs/reports/metrics_summary.json", "w") as f:
        json.dump(final_report, f, indent=2)

    # Generate Model Visualizations
    plot_loss_curves(history, "outputs/figures/model_1_loss_curves.png")
    plot_accuracy_curves(history, "outputs/figures/model_2_accuracy_curves.png")
    plot_confusion_matrix_heatmap(y_test, mlp_test_probs, threshold=0.5, save_path="outputs/figures/model_3_confusion_matrix.png")
    plot_roc_curve_viz(y_test, mlp_test_probs, save_path="outputs/figures/model_4_roc_auc_curve.png")
    plot_precision_recall_viz(y_test, mlp_test_probs, save_path="outputs/figures/model_5_precision_recall_curve.png")
    plot_model_comparison_benchmark(benchmark_dict, save_path="outputs/figures/model_6_baseline_vs_mlp_comparison.png")
    print("\nSaved Model Evaluation figures to outputs/figures/ (6 figures generated).")
    print("=" * 70)
    print("ONLINE SHOPPERS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
