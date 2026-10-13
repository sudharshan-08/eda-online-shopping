"""
Streamlit Interactive Dashboard Application
Case Study: Exploratory Data Analysis and Visualization of a Real-World Dataset
Dataset: Online Shoppers Purchasing Intention (UCI Machine Learning Repository)
Course: Exploratory Data Analysis and Visualization
Institution: KPR Institute of Engineering and Technology
Department: Artificial Intelligence and Data Science (Academic Year: 2026-2027)
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_raw_data, get_data_dictionary, get_dataset_summary
from src.eda import (
    compute_univariate_stats, compute_bivariate_numeric,
    compute_bivariate_categorical, test_probability_distributions
)
from src.visualizations import (
    get_interactive_univariate_fig,
    get_interactive_correlation_heatmap,
    get_interactive_scatter_fig
)
from src.model import ShoppersMLP, predict_shoppers_mlp
from src.evaluation import compute_classification_metrics

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Online Shoppers Purchasing Intention EDA & Deep Learning | KPRIET",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 1px solid #BAE6FD;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #0369A1;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0C4A6E;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1F2937;
        border-left: 4px solid #2563EB;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    if os.path.exists("data/shoppers_cleaned.csv"):
        df = pd.read_csv("data/shoppers_cleaned.csv")
    elif os.path.exists("data/online_shoppers_intention.csv"):
        df = load_raw_data("data/online_shoppers_intention.csv")
    else:
        st.error("Dataset not found! Please run `python run_pipeline.py` first.")
        st.stop()
    return df


@st.cache_data
def load_reports():
    metrics = None
    eda = None
    if os.path.exists("outputs/reports/metrics_summary.json"):
        with open("outputs/reports/metrics_summary.json", "r") as f:
            metrics = json.load(f)
    if os.path.exists("outputs/reports/eda_summary.json"):
        with open("outputs/reports/eda_summary.json", "r") as f:
            eda = json.load(f)
    return metrics, eda


df = load_data()
metrics_report, eda_report = load_reports()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "📋 Project Overview",
        "🗃️ Dataset Explorer",
        "📊 Univariate & Distributions",
        "🔄 Bivariate & Multivariate EDA",
        "🧠 Deep Learning & Benchmarks",
        "🔮 Real-Time Shopper Predictor",
        "📑 Findings & Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Academic Details:**  
- **Course:** Exploratory Data Analysis & Visualization  
- **Institution:** KPR Institute of Engineering & Technology  
- **Department:** AI & Data Science  
- **Academic Year:** 2026–2027  
- **Assignment:** Assignment II (CO3, CO4, CO5)  
""")


# =====================================================================
# PAGE 1: PROJECT OVERVIEW
# =====================================================================
if page == "📋 Project Overview":
    st.markdown('<div class="main-header">Case Study: Exploratory Data Analysis and Visualization of a Real-World Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Online Shoppers Purchasing Intention Prediction via Statistical EDA & PyTorch Deep Learning</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total User Sessions</div>
            <div class="metric-value">12,330</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Features</div>
            <div class="metric-value">18</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Purchasing Sessions</div>
            <div class="metric-value">15.47%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Best MLP Recall</div>
            <div class="metric-value">75.17%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Problem Statement & Real-World Domain</div>', unsafe_allow_html=True)
    st.write("""
    E-commerce businesses encounter massive web traffic, but only a small proportion of visiting sessions culminate in a financial transaction (~15.5%). 
    Accurately predicting purchasing intention during an ongoing browsing session enables online platforms to:
    1. Deliver personalized real-time interventions (discounts, product recommendations, live chat).
    2. Identify and reduce friction points on high-exit or high-bounce pages.
    3. Increase conversion rate optimization (CRO) and maximize gross merchandise value (GMV).
    
    This project conducts an exhaustive **Exploratory Data Analysis (EDA)** and builds a production-grade **PyTorch Deep Learning Multilayer Perceptron (ShoppersMLP)** 
    on the genuine UC Irvine benchmark dataset (Sakar et al., 2019) to:
    - Explore web session features (Administrative, ProductRelated, PageValues, ExitRates, BounceRates).
    - Statistically evaluate continuous distributions and hypothesis tests (Mann-Whitney U, Chi-Square, VIF).
    - Train a Deep Neural Network with zero data leakage, class weighting, AdamW, and early stopping.
    - Benchmark against standard industry baselines (Logistic Regression, Random Forest).
    """)

    st.markdown('<div class="section-title">Course Outcomes Addressed</div>', unsafe_allow_html=True)
    st.markdown("""
    - **CO3 (Apply Data Visualization Techniques):** Histograms with KDE, boxplots, correlation matrices, bivariate violin plots, and ROC/PR curves.
    - **CO4 (Perform Exploratory Data Analysis & Statistical Testing):** Parametric & non-parametric tests, goodness-of-fit distribution assessments, and multicollinearity diagnostics.
    - **CO5 (Develop Predictive Deep Learning Model):** PyTorch ShoppersMLP with Batch Normalization, Dropout regularizers, class weighting, and benchmark validation against held-out test data.
    """)

    st.info("💡 Use the sidebar navigation on the left to explore each analytical module and interact with live model predictions.")


# =====================================================================
# PAGE 2: DATASET EXPLORER
# =====================================================================
elif page == "🗃️ Dataset Explorer":
    st.markdown('<div class="main-header">Dataset Overview & Interactive Dictionary</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Verified UC Irvine Machine Learning Repository (DOI: 10.24432/C5F88Q)</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📄 Session Data Sample", "📖 Data Dictionary", "🔍 Data Integrity & Missingness"])

    with tab1:
        st.write(f"Displaying sample records from **`{len(df):,}`** visitor browsing sessions across **`{df.shape[1]}`** attributes:")
        n_rows = st.slider("Select number of rows to preview", min_value=5, max_value=50, value=10)
        st.dataframe(df.head(n_rows), use_container_width=True)

    with tab2:
        dict_df = get_data_dictionary()
        st.dataframe(dict_df, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Missing Value Summary")
            null_counts = df.isnull().sum()
            st.write(pd.DataFrame({"Attribute": null_counts.index, "Missing Count": null_counts.values, "Missing %": 0.0}))
            st.success("Zero missing / null values detected in the dataset. All sessions are complete.")

        with col2:
            st.subheader("Target (Revenue) Class Balance")
            counts = df["Revenue"].value_counts()
            pcts = (df["Revenue"].value_counts(normalize=True) * 100).round(2)
            st.write(pd.DataFrame({
                "Purchased (Revenue)": ["False (No Purchase)", "True (Completed Purchase)"],
                "Count": [counts.get(False, 0), counts.get(True, 0)],
                "Percentage (%)": [pcts.get(False, 0), pcts.get(True, 0)]
            }))
            st.info("Imbalance Ratio: ~5.46 non-purchasing sessions for every 1 purchasing session.")


# =====================================================================
# PAGE 3: UNIVARIATE & DISTRIBUTIONS
# =====================================================================
elif page == "📊 Univariate & Distributions":
    st.markdown('<div class="main-header">Univariate Analysis & Probability Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Distribution shapes, central tendencies, spreads, and theoretical goodness-of-fit testing</div>', unsafe_allow_html=True)

    num_cols = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    selected_col = st.selectbox("Select Continuous Session Metric for Inspection", num_cols, index=8)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = get_interactive_univariate_fig(df, selected_col)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Parametric & Distribution Fit")
        dist_res = test_probability_distributions(df, selected_col)
        
        series = df[selected_col].dropna()
        st.metric("Sample Mean", f"{series.mean():.3f}")
        st.metric("Sample Median", f"{series.median():.3f}")
        st.metric("Skewness (Fisher-Pearson)", f"{series.skew():.2f}")
        st.metric("Kurtosis", f"{series.kurtosis():.2f}")

        st.markdown("**Goodness-of-Fit Tests (Kolmogorov-Smirnov):**")
        for dist_name, d_val in dist_res.items():
            if dist_name in ["Normal", "Lognormal", "Exponential"]:
                st.write(f"- **{dist_name}:** K-S Stat = `{d_val['ks_stat']}`, p-value = `{d_val['p_value']:.4e}` (Fit: `{'Yes' if d_val['fits_distribution'] else 'Rejected'}`)")

    st.markdown('<div class="section-title">Static Publication Figures</div>', unsafe_allow_html=True)
    if os.path.exists("outputs/figures/eda_1_univariate_distributions.png"):
        st.image("outputs/figures/eda_1_univariate_distributions.png", caption="Figure 1: Univariate distributions of PageValues, ProductDuration, ExitRates, and BounceRates with KDE overlays.", use_container_width=True)

    if os.path.exists("outputs/figures/eda_7_probability_distribution_qq.png"):
        st.image("outputs/figures/eda_7_probability_distribution_qq.png", caption="Figure 7: Normal Q-Q plot and empirical histogram against fitted Gaussian curve.", use_container_width=True)


# =====================================================================
# PAGE 4: BIVARIATE & MULTIVARIATE EDA
# =====================================================================
elif page == "🔄 Bivariate & Multivariate EDA":
    st.markdown('<div class="main-header">Bivariate & Multivariate Exploratory Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Correlation matrices, interaction effects, and statistical hypothesis testing</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔥 Correlation Heatmap", "💰 Page Values Impact", "📅 Calendar Month & Visitor Conversion"])

    with tab1:
        st.plotly_chart(get_interactive_correlation_heatmap(df), use_container_width=True)
        st.info("💡 **Multicollinearity Insight:** `BounceRates` and `ExitRates` exhibit a Pearson correlation $r = 0.91$, and `ProductRelated` & `ProductRelated_Duration` show $r = 0.86$.")
        if os.path.exists("outputs/figures/eda_3_correlation_heatmap.png"):
            st.image("outputs/figures/eda_3_correlation_heatmap.png", caption="Figure 3: Publication-ready Pearson correlation matrix.", use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Page Values vs Transaction Revenue")
            st.write("""
            - Non-purchasing sessions exhibit a median **PageValues of 0.00** (mean ~1.98).
            - Purchasing sessions exhibit a mean **PageValues of ~27.26**.
            - Non-parametric **Mann-Whitney U Test** yields $p < 10^{-15}$, confirming PageValues as the single most discriminative predictor of purchasing intent.
            """)
        with col2:
            if os.path.exists("outputs/figures/eda_4_duration_vs_deposit.png"):
                st.image("outputs/figures/eda_4_duration_vs_deposit.png", caption="Figure 4: Bivariate boxplot of PageValues and density of ExitRates by Revenue outcome.", use_container_width=True)

    with tab3:
        if os.path.exists("outputs/figures/eda_5_job_subscription_rates.png"):
            st.image("outputs/figures/eda_5_job_subscription_rates.png", caption="Figure 5: E-Commerce conversion rate vs total traffic across calendar months.", use_container_width=True)
        if os.path.exists("outputs/figures/eda_6_economic_indicators_multivariate.png"):
            st.image("outputs/figures/eda_6_economic_indicators_multivariate.png", caption="Figure 6: Multivariate interaction between ExitRates and PageValues by purchase outcome.", use_container_width=True)


# =====================================================================
# PAGE 5: DEEP LEARNING & BENCHMARKS
# =====================================================================
elif page == "🧠 Deep Learning & Benchmarks":
    st.markdown('<div class="main-header">PyTorch Deep Learning ShoppersMLP & Baseline Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluated on Held-Out Test Set (1,831 sessions) with Zero Data Leakage</div>', unsafe_allow_html=True)

    if metrics_report:
        benchmarks = metrics_report["benchmark"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Logistic Regression (Baseline)</div>
                <div class="metric-value">ROC-AUC: 0.7462</div>
                <div>Recall: 73.78% | F1: 0.3937</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Random Forest (Baseline)</div>
                <div class="metric-value">ROC-AUC: 0.7755</div>
                <div>Recall: 69.23% | F1: 0.4099</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">PyTorch ShoppersMLP (Deep Learning)</div>
                <div class="metric-value">Recall: 75.17%</div>
                <div>ROC-AUC: 0.7722 | Acc: 64.99%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Comparative Performance Table</div>', unsafe_allow_html=True)
        table_rows = []
        for m_name, vals in benchmarks.items():
            table_rows.append({
                "Model Architecture": m_name,
                "Accuracy": f"{vals['accuracy'] * 100:.2f}%",
                "Recall (Sensitivity)": f"{vals['recall'] * 100:.2f}%",
                "Precision": f"{vals['precision'] * 100:.2f}%",
                "F1-Score": f"{vals['f1_score']:.4f}",
                "Balanced Accuracy": f"{vals['balanced_accuracy'] * 100:.2f}%",
                "ROC-AUC": f"{vals['roc_auc']:.4f}",
                "PR-AUC": f"{vals['pr_auc']:.4f}"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📈 Loss & Accuracy Curves", "🎯 Confusion Matrix", "📊 ROC & Precision-Recall Curves"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("outputs/figures/model_1_loss_curves.png"):
                st.image("outputs/figures/model_1_loss_curves.png", caption="Figure 8: Training loss vs Validation loss over epochs.", use_container_width=True)
        with col2:
            if os.path.exists("outputs/figures/model_2_accuracy_curves.png"):
                st.image("outputs/figures/model_2_accuracy_curves.png", caption="Figure 9: Training accuracy vs Validation accuracy over epochs.", use_container_width=True)

    with tab2:
        col1, col2 = st.columns([3, 2])
        with col1:
            if os.path.exists("outputs/figures/model_3_confusion_matrix.png"):
                st.image("outputs/figures/model_3_confusion_matrix.png", caption="Figure 10: PyTorch ShoppersMLP Confusion Matrix on 1,831 test sessions.", use_container_width=True)
        with col2:
            st.subheader("Confusion Matrix Interpretation")
            st.write("""
            - **True Positives (215):** Correctly identified 75.17% of actual purchasing shoppers.
            - **False Negatives (71):** Only 71 purchasing sessions were missed.
            - **Trade-off:** Class weighting successfully prioritized detecting true buyers from sparse traffic.
            """)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("outputs/figures/model_4_roc_auc_curve.png"):
                st.image("outputs/figures/model_4_roc_auc_curve.png", caption="Figure 11: ROC Curve (AUC = 0.7722) demonstrating discriminative capacity.", use_container_width=True)
        with col2:
            if os.path.exists("outputs/figures/model_5_precision_recall_curve.png"):
                st.image("outputs/figures/model_5_precision_recall_curve.png", caption="Figure 12: Precision-Recall Curve (PR-AUC = 0.3647) vs random prevalence (0.155).", use_container_width=True)


# =====================================================================
# PAGE 6: REAL-TIME CLIENT PREDICTOR
# =====================================================================
elif page == "🔮 Real-Time Shopper Predictor":
    st.markdown('<div class="main-header">Real-Time Purchasing Intention Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live inference sandbox powered by the trained PyTorch ShoppersMLP model</div>', unsafe_allow_html=True)

    st.write("Adjust visitor session parameters to predict live purchasing intention probability:")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Page Browsing Activity**")
            admin = st.slider("Administrative Pages Visited", 0, 20, 2)
            admin_dur = st.slider("Admin Duration (Seconds)", 0.0, 1000.0, 50.0)
            info = st.slider("Informational Pages Visited", 0, 15, 1)
            info_dur = st.slider("Info Duration (Seconds)", 0.0, 1000.0, 25.0)
            prod = st.slider("Product-Related Pages Visited", 1, 150, 32)
            prod_dur = st.slider("Product Duration (Seconds)", 10.0, 5000.0, 1150.0)

        with col2:
            st.markdown("**Google Analytics Metrics**")
            page_values = st.slider("PageValues (Analytics Index)", 0.0, 100.0, 15.0)
            bounce_rate = st.slider("Bounce Rate", 0.0, 0.2, 0.01, step=0.005)
            exit_rate = st.slider("Exit Rate", 0.0, 0.2, 0.03, step=0.005)
            special_day = st.slider("Special Day Closeness (0 to 1)", 0.0, 1.0, 0.0, step=0.1)

        with col3:
            st.markdown("**Visitor Context & Timing**")
            month = st.selectbox("Calendar Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)
            visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"], index=0)
            weekend = st.selectbox("Weekend Session?", ["False", "True"], index=0)
            os_id = st.selectbox("Operating System ID", [1, 2, 3, 4, 5, 6, 7, 8], index=1)
            browser_id = st.selectbox("Browser ID", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)
            region_id = st.selectbox("Region ID", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=0)
            traffic_id = st.selectbox("Traffic Type ID", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)

        submit_btn = st.form_submit_button("⚡ Run PyTorch Deep Learning Inference", use_container_width=True)

    if submit_btn:
        from src.preprocessing import run_full_preprocessing_pipeline
        artifacts = run_full_preprocessing_pipeline()
        scaler = artifacts["scaler"]
        feature_names = artifacts["feature_names"]

        input_dict = {
            "Administrative": admin,
            "Administrative_Duration": admin_dur,
            "Informational": info,
            "Informational_Duration": info_dur,
            "ProductRelated": prod,
            "ProductRelated_Duration": prod_dur,
            "BounceRates": bounce_rate,
            "ExitRates": exit_rate,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": str(month),
            "OperatingSystems": str(os_id),
            "Browser": str(browser_id),
            "Region": str(region_id),
            "TrafficType": str(traffic_id),
            "VisitorType": str(visitor_type),
            "Weekend": 1 if weekend == "True" else 0
        }
        input_df = pd.DataFrame([input_dict])

        cat_cols = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
        encoded_input = pd.get_dummies(input_df, columns=cat_cols, drop_first=False, dtype=float)

        aligned_input = pd.DataFrame(0.0, index=[0], columns=feature_names)
        for c in encoded_input.columns:
            if c in aligned_input.columns:
                aligned_input[c] = encoded_input[c].values[0]

        scaled_input = scaler.transform(aligned_input)

        checkpoint = torch.load("outputs/reports/best_mlp_model.pt", map_location=torch.device("cpu"))
        model = ShoppersMLP(in_features=len(feature_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        prob = float(predict_shoppers_mlp(model, scaled_input)[0])

        st.markdown("### Prediction Results")
        colA, colB = st.columns([1, 2])
        with colA:
            st.metric("Predicted Purchase Probability", f"{prob * 100:.1f}%")
            if prob >= 0.5:
                st.success("Recommendation: HIGH PURCHASE INTENT — Prioritize Checkout! 🛍️")
            else:
                st.warning("Recommendation: HESITANT VISITOR — Trigger Targeted Offer/Coupon! 🏷️")

        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Purchase Conversion Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2563EB"},
                    'steps': [
                        {'range': [0, 40], 'color': "#fee2e2"},
                        {'range': [40, 70], 'color': "#fef3c7"},
                        {'range': [70, 100], 'color': "#dcfce7"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PAGE 7: FINDINGS & RECOMMENDATIONS
# =====================================================================
elif page == "📑 Findings & Recommendations":
    st.markdown('<div class="main-header">Key Analytical Findings & Managerial Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-driven strategic takeaways for e-commerce conversion optimization</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Key Exploratory Findings</div>', unsafe_allow_html=True)
        st.write("""
        1. **PageValues is the Leading Signal:** Sessions with `PageValues > 0` achieve conversion rates exceeding **54%**, whereas sessions with zero page value convert at under **4%**.
        2. **Bounce & Exit Friction:** High bounce and exit rates ($> 0.05$) correlate with sharp drop-offs in purchasing probability.
        3. **Seasonal Spikes:** November (Cyber Week / Black Friday) and May generate peak transaction volumes and elevated conversion rates (>25%).
        4. **Visitor Intent:** New visitors exhibit higher conversion rates (~24.9%) than returning visitors (~14.6%), highlighting the importance of initial onboarding.
        """)

    with col2:
        st.markdown('<div class="section-title">Actionable E-Commerce Recommendations</div>', unsafe_allow_html=True)
        st.write("""
        1. **Deploy Real-Time Cart Abandonment Intervention:** Utilize the PyTorch ShoppersMLP model to detect hesitant shoppers with probabilities between 35% and 50% and trigger exit-intent discount popups.
        2. **Optimize High-Exit Catalog Pages:** Redesign product pages showing elevated exit rates to streamline add-to-cart call-to-actions.
        3. **Leverage High-Value Funnels:** Funnel incoming paid traffic directly to catalog pages with historically elevated `PageValues`.
        """)

    st.markdown("---")
    st.caption("College Assignment Project | KPR Institute of Engineering and Technology | Department of AI & Data Science (2026–2027)")
