"""
Exploratory Data Analysis (EDA) & Statistical Testing Module for Online Shoppers Dataset
Implements Univariate, Bivariate, Multivariate analysis, Probability Distribution testing,
and statistical hypothesis tests (Mann-Whitney U, Chi-Square, Cramer's V, VIF).
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple


def compute_univariate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rigorous univariate statistical metrics for all continuous numerical features:
    Mean, Median, Std Dev, Min, 25%, 50%, 75%, Max, IQR, Skewness, Kurtosis.
    """
    num_cols = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    records = []

    for col in num_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        mean = series.mean()
        median = series.median()
        std = series.std()
        skew = series.skew()
        kurt = series.kurtosis()

        records.append({
            "Feature": col,
            "Count": len(series),
            "Mean": round(mean, 3),
            "Median": round(median, 3),
            "Std_Dev": round(std, 3),
            "Min": round(series.min(), 3),
            "Q1_25%": round(q1, 3),
            "Q3_75%": round(q3, 3),
            "Max": round(series.max(), 3),
            "IQR": round(iqr, 3),
            "Skewness": round(skew, 3),
            "Kurtosis": round(kurt, 3)
        })

    return pd.DataFrame(records)


def compute_categorical_frequencies(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Computes frequency counts and percentage shares for categorical and discrete session features.
    """
    cat_cols = ["Month", "VisitorType", "Weekend", "OperatingSystems", "Browser", "Region", "TrafficType"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    freq_dict = {}

    for col in cat_cols:
        counts = df[col].value_counts()
        pcts = (df[col].value_counts(normalize=True) * 100).round(2)
        summary_df = pd.DataFrame({
            "Category": counts.index.astype(str),
            "Count": counts.values,
            "Percentage (%)": pcts.values
        })
        freq_dict[col] = summary_df

    return freq_dict


def test_probability_distributions(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Evaluates whether an e-commerce session continuous variable fits theoretical distributions:
    - Normal (Gaussian)
    - Lognormal
    - Exponential
    
    Uses Kolmogorov-Smirnov (K-S) goodness-of-fit test.
    """
    data = df[column].dropna().values
    data_pos = data[data > 0]

    results = {}

    # 1. Normal distribution test
    norm_params = stats.norm.fit(data)
    ks_norm_stat, ks_norm_p = stats.kstest(data, "norm", args=norm_params)
    results["Normal"] = {
        "params": {"mean": round(float(norm_params[0]), 3), "std": round(float(norm_params[1]), 3)},
        "ks_stat": round(float(ks_norm_stat), 4),
        "p_value": float(ks_norm_p),
        "fits_distribution": bool(ks_norm_p > 0.05)
    }

    # 2. Lognormal distribution test (for positive values)
    if len(data_pos) > 10:
        lognorm_params = stats.lognorm.fit(data_pos)
        ks_logn_stat, ks_logn_p = stats.kstest(data_pos, "lognorm", args=lognorm_params)
        results["Lognormal"] = {
            "params": {"s": round(float(lognorm_params[0]), 3), "scale": round(float(lognorm_params[2]), 3)},
            "ks_stat": round(float(ks_logn_stat), 4),
            "p_value": float(ks_logn_p),
            "fits_distribution": bool(ks_logn_p > 0.05)
        }

    # 3. Exponential distribution test
    if len(data_pos) > 10:
        expon_params = stats.expon.fit(data_pos)
        ks_exp_stat, ks_exp_p = stats.kstest(data_pos, "expon", args=expon_params)
        results["Exponential"] = {
            "params": {"scale": round(float(expon_params[1]), 3)},
            "ks_stat": round(float(ks_exp_stat), 4),
            "p_value": float(ks_exp_p),
            "fits_distribution": bool(ks_exp_p > 0.05)
        }

    # D'Agostino-Pearson omnibus test for normality
    if len(data) >= 20:
        stat, norm_p = stats.normaltest(data)
        results["DAgostino_Pearson_Normality"] = {
            "statistic": round(float(stat), 3),
            "p_value": float(norm_p),
            "is_normal": bool(norm_p > 0.05)
        }

    return results


def compute_bivariate_numeric(df: pd.DataFrame, target_col: str = "Revenue") -> pd.DataFrame:
    """
    Performs bivariate analysis between continuous session features and the binary target Revenue.
    Compares metrics for Non-Purchasers (Revenue=False) vs Purchasers (Revenue=True)
    and executes non-parametric Mann-Whitney U tests.
    """
    num_cols = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    records = []

    group_no = df[df[target_col] == False]
    group_yes = df[df[target_col] == True]

    for col in num_cols:
        v_no = group_no[col].dropna()
        v_yes = group_yes[col].dropna()

        u_stat, p_val = stats.mannwhitneyu(v_no, v_yes, alternative="two-sided")

        n1, n2 = len(v_no), len(v_yes)
        s1, s2 = v_no.std(), v_yes.std()
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        cohen_d = (v_yes.mean() - v_no.mean()) / pooled_std if pooled_std != 0 else 0.0

        records.append({
            "Feature": col,
            "Mean_NoPurchase": round(v_no.mean(), 3),
            "Median_NoPurchase": round(v_no.median(), 3),
            "Mean_Purchase": round(v_yes.mean(), 3),
            "Median_Purchase": round(v_yes.median(), 3),
            "Mann_Whitney_U": round(float(u_stat), 1),
            "p_value": float(p_val),
            "Significant_at_0.05": bool(p_val < 0.05),
            "Cohens_d": round(float(cohen_d), 3)
        })

    return pd.DataFrame(records)


def compute_bivariate_categorical(df: pd.DataFrame, target_col: str = "Revenue") -> List[Dict[str, Any]]:
    """
    Performs bivariate analysis between categorical attributes and the purchasing target Revenue.
    Executes Pearson's Chi-Square Test of Independence and calculates Cramer's V effect size.
    """
    cat_cols = ["Month", "VisitorType", "Weekend", "OperatingSystems", "Browser", "Region", "TrafficType"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    results = []

    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[target_col])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

        # Purchase conversion rate per category
        rates = (df.groupby(col)[target_col].apply(lambda s: s.mean()) * 100).round(2).to_dict()
        rates_str = {str(k): float(v) for k, v in rates.items()}

        results.append({
            "Feature": col,
            "Chi2_Statistic": round(float(chi2), 3),
            "Degrees_of_Freedom": int(dof),
            "p_value": float(p_val),
            "Statistically_Significant": bool(p_val < 0.05),
            "Cramers_V": round(float(cramer_v), 3),
            "Category_Conversion_Rates_%": rates_str
        })

    return results


def compute_correlation_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates both Pearson (linear) and Spearman (rank) correlation matrices across continuous features.
    """
    num_cols = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    num_df = df[[c for c in num_cols if c in df.columns]]
    pearson_corr = num_df.corr(method="pearson").round(3)
    spearman_corr = num_df.corr(method="spearman").round(3)
    return pearson_corr, spearman_corr


def check_multicollinearity_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Variance Inflation Factor (VIF) for continuous session metrics.
    Highlights strong collinearity between BounceRates and ExitRates.
    """
    from sklearn.linear_model import LinearRegression

    num_cols = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    df_clean = df[num_cols].dropna()
    vif_records = []

    for col in num_cols:
        y_col = df_clean[col]
        X_other = df_clean[[c for c in num_cols if c != col]]
        reg = LinearRegression().fit(X_other, y_col)
        r2 = reg.score(X_other, y_col)
        vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else 999.0
        vif_records.append({
            "Feature": col,
            "R_Squared": round(r2, 3),
            "VIF": round(vif, 2),
            "Multicollinearity_Risk": "High" if vif > 10 else ("Moderate" if vif > 5 else "Low")
        })

    return pd.DataFrame(vif_records).sort_values(by="VIF", ascending=False)
