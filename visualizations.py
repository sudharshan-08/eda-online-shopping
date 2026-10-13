"""
Visualization Module for Online Shoppers Purchasing Intention Dataset
Generates high-resolution publication-quality Matplotlib/Seaborn figures (saved to outputs/figures/)
and interactive Plotly figures for the Streamlit dashboard.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List

# Set clean publication aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


# =====================================================================
# STATIC MATPLOTLIB / SEABORN VISUALIZATIONS (FOR REPORTS & FIGURES)
# =====================================================================

def plot_univariate_distributions(df: pd.DataFrame, save_path: str = "outputs/figures/eda_1_univariate_distributions.png") -> None:
    """
    Visualization 1: Univariate Distributions of Key Continuous E-Commerce Metrics
    Plots histograms with Kernel Density Estimation (KDE) overlays for PageValues,
    ProductRelated_Duration, ExitRates, and BounceRates.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Univariate Distributions of Key Browsing Metrics (with KDE Overlays)", fontsize=15, fontweight="bold", y=0.98)

    features = [
        ("PageValues", "Page Value (Google Analytics Index)", "#1f77b4"),
        ("ProductRelated_Duration", "Product Browsing Duration (Seconds)", "#ff7f0e"),
        ("ExitRates", "Page Exit Rate (%)", "#2ca02c"),
        ("BounceRates", "Page Bounce Rate (%)", "#d62728")
    ]

    for ax, (col, label, color) in zip(axes.flatten(), features):
        sns.histplot(df[col], kde=True, ax=ax, color=color, bins=35, stat="density", alpha=0.55)
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="black", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.2f}")
        ax.set_title(f"Distribution of {col}", fontsize=12, fontweight="semibold")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_outlier_boxplots(df: pd.DataFrame, save_path: str = "outputs/figures/eda_2_outlier_boxplots.png") -> None:
    """
    Visualization 2: Boxplots for E-Commerce Session Metrics Outlier Detection (Tukey IQR Inspection)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_cols = [
        "Administrative_Duration", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Outlier Analysis via Boxplots Across Session Features (Tukey 1.5x IQR Whiskers)", fontsize=15, fontweight="bold", y=0.98)

    for ax, col in zip(axes.flatten(), num_cols):
        sns.boxplot(y=df[col], ax=ax, color="#4c72b0", flierprops={"marker": "o", "markersize": 3, "alpha": 0.3, "color": "darkred"})
        ax.set_title(col, fontsize=11, fontweight="semibold")
        ax.set_ylabel("Observed Value", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = "outputs/figures/eda_3_correlation_heatmap.png") -> None:
    """
    Visualization 3: Pearson Correlation Heatmap for Continuous E-Commerce Metrics
    Identifies high multicollinearity between BounceRates & ExitRates, and ProductRelated & Duration.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_cols = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    corr = df[num_cols].corr(method="pearson")

    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"}
    )
    plt.title("Pearson Correlation Matrix of Continuous Web Browsing Features", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pagevalues_vs_revenue(df: pd.DataFrame, save_path: str = "outputs/figures/eda_4_duration_vs_deposit.png") -> None:
    """
    Visualization 4: Bivariate Relationship — PageValues & ExitRates vs Transaction Outcome (Revenue)
    Displays violin plots and log-density distributions.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Bivariate Analysis: Page Values and Exit Rates vs Purchasing Intention (Revenue)", fontsize=15, fontweight="bold", y=1.02)

    # Box/Violin plot of PageValues by Revenue
    df_plot = df.copy()
    df_plot["Revenue_Label"] = df_plot["Revenue"].map({False: "No Purchase (False)", True: "Purchased (True)"})

    sns.boxplot(
        x="Revenue_Label", y="PageValues", data=df_plot, ax=ax1,
        palette={"No Purchase (False)": "#e74c3c", "Purchased (True)": "#2ecc71"}
    )
    ax1.set_title("Distribution of PageValues by Purchasing Status", fontsize=12, fontweight="semibold")
    ax1.set_xlabel("Transaction Outcome (Revenue)", fontsize=11)
    ax1.set_ylabel("PageValues (Google Analytics Metric)", fontsize=11)

    # KDE density plot of ExitRates by Revenue
    sns.kdeplot(
        df["ExitRates"][df["Revenue"] == False], ax=ax2, label="No Purchase (False)",
        color="#e74c3c", fill=True, alpha=0.35, linewidth=2
    )
    sns.kdeplot(
        df["ExitRates"][df["Revenue"] == True], ax=ax2, label="Purchased (True)",
        color="#2ecc71", fill=True, alpha=0.35, linewidth=2
    )
    ax2.set_title("Density Comparison of Page Exit Rates by Outcome", fontsize=12, fontweight="semibold")
    ax2.set_xlabel("Exit Rates", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.legend(title="Revenue Outcome", loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_month_and_visitor_conversion(df: pd.DataFrame, save_path: str = "outputs/figures/eda_5_job_subscription_rates.png") -> None:
    """
    Visualization 5: Categorical Analysis — Purchasing Conversion Rates Across Calendar Months & Visitor Types
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    existing_months = [m for m in month_order if m in df["Month"].unique()]

    month_summary = df.groupby("Month")["Revenue"].agg(
        Total="count",
        Purchased=lambda s: s.sum(),
        ConversionRate=lambda s: s.mean() * 100
    ).reindex(existing_months).reset_index()

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    bars = ax1.bar(
        month_summary["Month"], month_summary["ConversionRate"],
        color="#3498db", alpha=0.85, width=0.55, label="Conversion Rate (%)"
    )
    line = ax2.plot(
        month_summary["Month"], month_summary["Total"],
        color="#e67e22", marker="o", linewidth=2.5, label="Total Session Traffic"
    )

    ax1.set_title("E-Commerce Purchase Conversion Rate vs Total Session Traffic by Month", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Calendar Month", fontsize=11)
    ax1.set_ylabel("Purchase Conversion Rate (%)", color="#2980b9", fontsize=11)
    ax2.set_ylabel("Total Browsing Sessions", color="#d35400", fontsize=11)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_exitrates_vs_pagevalues(df: pd.DataFrame, save_path: str = "outputs/figures/eda_6_economic_indicators_multivariate.png") -> None:
    """
    Visualization 6: Multivariate Analysis — ExitRates vs PageValues Colored by Revenue Outcome
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(11, 7))

    sns.scatterplot(
        x="ExitRates", y="PageValues", hue="Revenue", data=df,
        palette={False: "#95a5a6", True: "#27ae60"},
        alpha=0.55, s=30, edgecolor="none"
    )
    plt.title("Multivariate Interaction: Page Exit Rates vs Page Values by Purchase Outcome", fontsize=13, fontweight="bold")
    plt.xlabel("Exit Rates", fontsize=11)
    plt.ylabel("Page Values (Google Analytics Value Index)", fontsize=11)
    plt.legend(title="Revenue (Purchased)", loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distribution_fit(df: pd.DataFrame, column: str = "ProductRelated_Duration", save_path: str = "outputs/figures/eda_7_probability_distribution_qq.png") -> None:
    """
    Visualization 7: Probability Distribution Fit & Q-Q Plot Analysis
    Compares the empirical distribution of ProductRelated_Duration against theoretical Gaussian model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = df[column].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Probability Distribution Fit & Q-Q Plot Analysis for '{column}'", fontsize=14, fontweight="bold")

    # Histogram with fitted Gaussian curve
    sns.histplot(data, kde=False, stat="density", ax=ax1, color="#2980b9", alpha=0.5, bins=30)
    mu, std = stats.norm.fit(data)
    x_grid = np.linspace(data.min(), data.max(), 200)
    pdf_fitted = stats.norm.pdf(x_grid, mu, std)
    ax1.plot(x_grid, pdf_fitted, "r-", linewidth=2.5, label=f"Fitted Gaussian (μ={mu:.1f}, σ={std:.1f})")
    ax1.set_title(f"Empirical Histogram vs Fitted Gaussian", fontsize=11, fontweight="semibold")
    ax1.set_xlabel(column, fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.legend(loc="upper right")

    # Q-Q Plot against theoretical standard normal
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title("Quantile-Quantile (Q-Q) Plot vs Standard Normal", fontsize=11, fontweight="semibold")
    ax2.get_lines()[0].set_markersize(4.0)
    ax2.get_lines()[0].set_color("#2c3e50")
    ax2.get_lines()[1].set_color("red")
    ax2.get_lines()[1].set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_eda_visualizations(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Generates and saves all static EDA figures in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    plot_univariate_distributions(df, os.path.join(output_dir, "eda_1_univariate_distributions.png"))
    plot_outlier_boxplots(df, os.path.join(output_dir, "eda_2_outlier_boxplots.png"))
    plot_correlation_heatmap(df, os.path.join(output_dir, "eda_3_correlation_heatmap.png"))
    plot_pagevalues_vs_revenue(df, os.path.join(output_dir, "eda_4_duration_vs_deposit.png"))
    plot_month_and_visitor_conversion(df, os.path.join(output_dir, "eda_5_job_subscription_rates.png"))
    plot_exitrates_vs_pagevalues(df, os.path.join(output_dir, "eda_6_economic_indicators_multivariate.png"))
    plot_distribution_fit(df, column="ProductRelated_Duration", save_path=os.path.join(output_dir, "eda_7_probability_distribution_qq.png"))


# =====================================================================
# INTERACTIVE PLOTLY VISUALIZATIONS (FOR STREAMLIT DASHBOARD)
# =====================================================================

def get_interactive_univariate_fig(df: pd.DataFrame, column: str) -> go.Figure:
    """Returns an interactive Plotly histogram and boxplot for Streamlit."""
    df_plot = df.copy()
    df_plot["Purchased"] = df_plot["Revenue"].map({False: "No Purchase", True: "Purchase"})
    
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(
            df_plot, x=column, color="Purchased", marginal="box",
            barmode="overlay", opacity=0.6,
            color_discrete_map={"No Purchase": "#e74c3c", "Purchase": "#2ecc71"},
            title=f"Interactive Distribution of {column} by Purchasing Outcome"
        )
    else:
        fig = px.histogram(
            df_plot, x=column, color="Purchased", barmode="group",
            color_discrete_map={"No Purchase": "#e74c3c", "Purchase": "#2ecc71"},
            title=f"Category Frequency of {column} by Purchasing Outcome"
        )
    fig.update_layout(template="plotly_white", font=dict(family="sans-serif", size=12))
    return fig


def get_interactive_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Returns an interactive Plotly correlation heatmap."""
    num_cols = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Interactive Pearson Correlation Matrix (Web Session Features)"
    )
    fig.update_layout(template="plotly_white")
    return fig


def get_interactive_scatter_fig(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Returns an interactive Plotly scatter plot with purchase coloring."""
    sample_df = df.sample(min(3000, len(df)), random_state=42).copy()
    sample_df["Purchased"] = sample_df["Revenue"].map({False: "No Purchase", True: "Purchase"})
    fig = px.scatter(
        sample_df, x=x_col, y=y_col, color="Purchased",
        color_discrete_map={"No Purchase": "#95a5a6", "Purchase": "#27ae60"},
        opacity=0.6,
        title=f"Scatter Plot: {x_col} vs {y_col} (Colored by Purchase Outcome)"
    )
    fig.update_layout(template="plotly_white")
    return fig
