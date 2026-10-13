# Case Study: Exploratory Data Analysis and Visualization of a Real-World Dataset
## Online Shoppers Purchasing Intention Prediction via Statistical EDA & PyTorch Deep Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B.svg)](https://streamlit.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests Passing](https://img.shields.io/badge/pytest-8%20passed-brightgreen.svg)]()

---

### Academic Information
- **Course:** Exploratory Data Analysis and Visualization (21AD401)
- **Institution:** KPR Institute of Engineering and Technology, Coimbatore
- **Department:** Department of Artificial Intelligence and Data Science
- **Academic Year:** 2026–2027 (Semester VI)
- **Assignment:** Assignment II (Total Marks: 40)
- **Course Outcomes Satisfied:**
  - **CO3:** Apply data visualization techniques to uncover patterns and relationships.
  - **CO4:** Perform exploratory data analysis, probability distribution fitting, and hypothesis testing.
  - **CO5:** Develop, train, and evaluate predictive machine learning / deep learning models.

---

## 1. Executive Summary & Problem Definition
In digital retail, user browsing telemetry contains critical behavioral signals that reveal customer purchase intent. However, the vast majority of e-commerce user visits (~84.5%) terminate without completing a purchase transaction.

This repository implements a production-grade data science project evaluating the genuine **UC Irvine Online Shoppers Purchasing Intention benchmark dataset** (12,330 records, 18 attributes). The system features:
1. **Deduplication & IQR Winsorization Capping:** Eliminates session duplicates while preserving high-activity purchasing shoppers.
2. **Zero-Data-Leakage Splitting & Robust Scaling:** Stratified Train (70%), Validation (15%), and Test (15%) partitions with scaling fitted strictly on training data.
3. **Statistical Hypothesis Testing:** Non-parametric Mann-Whitney U tests ($p < 10^{-15}$), Chi-Square tests of independence ($\chi^2$), Cramer's V effect sizes, and Variance Inflation Factor (VIF) collinearity diagnostics.
4. **PyTorch Deep Learning Tabular Classifier (`ShoppersMLP`):** Multi-layer perceptron with Batch Normalization, Dropout regularization, class-weighted Binary Cross-Entropy loss ($\text{pos\_weight} \approx 5.46$), AdamW optimization, and early stopping.
5. **Rigorous Baseline Benchmarking:** Direct comparative evaluation against Logistic Regression and Random Forest.
6. **Interactive Streamlit Web Dashboard (`app.py`):** Real-time interactive data explorer, live filterable charts, and an inference sandbox with a live probability gauge.
7. **Publication-Grade PDF Report (`docs/project_report.pdf`):** Complete academic documentation formatted with ReportLab.

---

## 2. Dataset Attribution & Characteristics
- **Dataset Name:** Online Shoppers Purchasing Intention Dataset (`online_shoppers_intention.csv`)
- **Source:** UC Irvine Machine Learning Repository
- **Direct Link:** [https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **DOI:** [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)
- **Citation:** C. O. Sakar, S. O. Polat, M. Katircioglu, and Y. Kastro. *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications, 31(10):6893-6908, October 2019.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

### Key Metrics
- **Total Records:** 12,330 visitor sessions
- **Total Attributes:** 18 (10 continuous numerical, 7 discrete/categorical, 1 binary target `Revenue`)
- **Target Distribution:** `False` (No Purchase) = 10,422 (84.53%) | `True` (Purchase) = 1,908 (15.47%)
- **Imbalance Ratio:** 5.46 : 1

*Full field descriptions, scales, and data types are catalogued in [`data/README.md`](data/README.md).*

---

## 3. Project Architecture

```text
eda/
├── app.py                     # Interactive Streamlit Web Dashboard
├── run_pipeline.py            # Master pipeline runner (EDA, Training, Evaluation)
├── requirements.txt           # Verified Python package dependencies
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Version control exclusions
│
├── data/
│   ├── README.md              # Data dictionary, provenance, and license info
│   ├── online_shoppers_intention.csv # Genuine raw dataset (12,330 rows x 18 cols)
│   └── shoppers_cleaned.csv   # Preprocessed and cleaned dataset
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb # End-to-end Jupyter analysis workflow
│   └── generate_notebook.py   # Automated notebook builder
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading, validation, integrity audits
│   ├── preprocessing.py       # Deduplication, IQR capping, encoding, leakage-free scaling
│   ├── eda.py                 # Univariate, bivariate, hypothesis tests, VIF
│   ├── visualizations.py      # Publication-grade Matplotlib & Plotly figures
│   ├── model.py               # PyTorch ShoppersMLP architecture & baseline models
│   └── evaluation.py          # Metrics calculation and model evaluation charts
│
├── outputs/
│   ├── figures/               # 13 high-resolution 300 DPI generated charts
│   │   ├── eda_1_univariate_distributions.png
│   │   ├── eda_2_outlier_boxplots.png
│   │   ├── eda_3_correlation_heatmap.png
│   │   ├── eda_4_duration_vs_deposit.png
│   │   ├── eda_5_job_subscription_rates.png
│   │   ├── eda_6_economic_indicators_multivariate.png
│   │   ├── eda_7_probability_distribution_qq.png
│   │   ├── model_1_loss_curves.png
│   │   ├── model_2_accuracy_curves.png
│   │   ├── model_3_confusion_matrix.png
│   │   ├── model_4_roc_auc_curve.png
│   │   ├── model_5_precision_recall_curve.png
│   │   └── model_6_baseline_vs_mlp_comparison.png
│   └── reports/
│       ├── best_mlp_model.pt  # Saved PyTorch model checkpoint
│       ├── eda_summary.json   # Computed statistical and test outputs
│       └── metrics_summary.json # Test benchmark metrics
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # Automated unit & integration tests (pytest)
│
└── docs/
    ├── project_report.md      # Full 15-section academic report (Markdown)
    ├── project_report.pdf     # Compiled academic PDF report (ReportLab)
    └── generate_pdf.py        # Automated PDF compilation script
```

---

## 4. Quantitative Results & Model Benchmarking

Evaluated strictly on the **1,831 held-out test sessions** (1,545 non-purchasers, 286 purchasers):

| Model Architecture | Accuracy | Recall (Sensitivity) | Precision | F1-Score | Balanced Accuracy | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 64.50% | 73.78% | 26.84% | 0.3937 | 68.28% | 0.7462 | 0.3255 |
| **Random Forest (Baseline)** | **68.87%** | 69.23% | **29.12%** | **0.4099** | 69.02% | **0.7755** | **0.3762** |
| **PyTorch ShoppersMLP (Deep Learning)** | 64.99% | **75.17%** | 27.39% | 0.4015 | **69.14%** | 0.7722 | 0.3647 |

### Key Trade-Off Insights
- **PyTorch ShoppersMLP achieves the highest Recall (75.17%)**, correctly identifying **215 out of 286 actual purchasing sessions**, minimizing missed conversion opportunities.
- **Random Forest delivers higher Precision (29.12%) and F1-score (0.4099)**, ideal when promotional incentive costs per shopper are high.
- Both models vastly outperform random guessing, confirming the utility of class-weighted loss functions and non-linear feature representations.

---

## 5. Installation & Execution Guide

### 5.1 Prerequisites
- Python 3.10 or 3.11
- Git

### 5.2 Environment Setup
```bash
git clone https://github.com/<your-username>/eda-visualization-project.git
cd eda-visualization-project

# Install dependencies
pip install -r requirements.txt
```

### 5.3 Step-by-Step Execution Commands

1. **Run Full Pipeline (Data Preprocessing, EDA, Model Training, Figure Generation):**
   ```bash
   python run_pipeline.py
   ```

2. **Execute Automated Test Suite:**
   ```bash
   python -m pytest tests/ -v
   ```

3. **Launch the Streamlit Interactive Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *Access the dashboard in your browser at `http://localhost:8501`.*

4. **Compile the Academic PDF Report:**
   ```bash
   python docs/generate_pdf.py
   ```

5. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/exploratory_data_analysis.ipynb
   ```

---

## 6. Key Business Takeaways & Strategic Recommendations

1. **PageValues is the Primary Revenue Indicator:** Sessions with `PageValues > 0` achieve a conversion rate of **54.2%**, compared to <4% for sessions with zero PageValues. Marketing traffic should be directly funneled into high-PageValue product paths.
2. **Catalog Page Friction Reduction:** Product pages exhibiting exit rates $> 0.05$ require immediate UX redesign (e.g. simplified variant selection, sticky checkout buttons).
3. **Capitalize on November Surge:** November conversion yield (25.4%) doubles the rest of the year. Marketing budgets should be heavily weighted towards Q4 holiday periods.
4. **Deploy Real-Time Dynamic Incentives:** Using ShoppersMLP to flag wavering shoppers (predicted purchase probability 35%–50%) enables real-time dynamic discounting to nudge them to checkout.

---

## 7. Submission Guide for GitHub

To push this repository to your personal GitHub account:

1. **Create a New Repository on GitHub:**
   - Log into [GitHub](https://github.com).
   - Click **New Repository** (e.g., `online-shoppers-eda-deep-learning`).
   - Do **NOT** initialize with a README, license, or .gitignore (these are already configured locally).

2. **Link Local Repository & Push:**
   Open terminal in `c:\Users\User\Desktop\eda` and run:
   ```bash
   git add .
   git commit -m "feat: complete overhaul to online shoppers purchasing intention dataset"
   git branch -M main
   git remote add origin https://github.com/<YOUR-GITHUB-USERNAME>/<REPO-NAME>.git
   git push -u origin main
   ```

---

## 8. Academic Integrity & Ethics Statement
All statistical analyses, machine learning models, and figures in this repository are computed directly from the genuine UC Irvine Online Shoppers Purchasing Intention dataset. No synthetic records or fabricated metrics are presented.

*Department of Artificial Intelligence and Data Science | KPR Institute of Engineering and Technology (2026–2027)*
