# CASE STUDY: EXPLORATORY DATA ANALYSIS AND VISUALIZATION OF A REAL-WORLD DATASET
## Real-Time Online Shoppers Purchasing Intention Prediction via Statistical EDA & PyTorch Deep Learning

---

### Academic Information & Title Page
- **Course Title:** Exploratory Data Analysis and Visualization
- **Course Code:** 21AD401 / EDA&V
- **Institution:** KPR Institute of Engineering and Technology, Coimbatore, Tamil Nadu
- **Department:** Department of Artificial Intelligence and Data Science
- **Academic Year:** 2026–2027
- **Semester:** VI
- **Assignment:** Assignment II (Total Marks: 40)
- **Course Outcomes Satisfied:** CO3 (Data Visualization), CO4 (Exploratory Data Analysis & Hypothesis Testing), CO5 (Predictive Deep Learning Modeling)
- **Student Name:** [Student Name / Register Number Placeholder]
- **Date of Submission:** October 2026

---

## 1. Abstract
In modern electronic commerce, understanding visitor browsing dynamics is critical for conversion rate optimization and personalized shopping experiences. However, the vast majority of e-commerce user sessions (~84.5%) terminate without completing a purchase transaction. This case study conducts an exhaustive exploratory analysis and predictive modeling workflow on the genuine UC Irvine Online Shoppers Purchasing Intention benchmark dataset, comprising 12,330 multi-session records across 18 behavioral and technical attributes. We implement an automated preprocessing pipeline that resolves session duplicates, applies Tukey IQR Winsorization capping to mitigate extreme variance while retaining high-value purchasing sessions, and guarantees zero data leakage through strictly partitioned feature scaling. Non-parametric Mann-Whitney U tests ($p < 10^{-15}$) identify PageValues and ExitRates as primary determinants of purchase propensity. We design, optimize, and train a PyTorch Multilayer Perceptron (ShoppersMLP) with Batch Normalization, Dropout regularization, and class-weighted Binary Cross-Entropy loss ($\text{pos\_weight} \approx 5.46$). Evaluated on a held-out test set of 1,831 sessions, ShoppersMLP attains a recall of 75.17% and an ROC-AUC of 0.7722, accurately capturing purchasing intent and offering clear trade-offs against Random Forest and Logistic Regression baselines.

---

## 2. Introduction and Objectives

### 2.1 Background
Electronic retail platforms process massive volumes of browsing telemetry every second. The conversion rate—the fraction of browsing sessions culminating in a monetary purchase—typically hovers around 2% to 15% across global online retail. Traditional batch analytics review customer transactions post-hoc, precluding timely interventions. Developing predictive models capable of real-time purchasing intention prediction from ongoing session telemetry empowers e-commerce systems to trigger dynamic micro-incentives (e.g. customized discounts, free shipping vouchers, or chatbot assistance) to convert wavering visitors into buyers.

### 2.2 Project Objectives
1. **Explore E-Commerce Session Telemetry (CO3, CO4):** Characterize browsing duration, catalog interaction, Google Analytics metrics, and temporal factors via publication-grade statistical charts.
2. **Empirical Distribution Testing (CO4):** Statistically assess continuous variables (e.g., `PageValues`, `ProductRelated_Duration`, `ExitRates`) against theoretical Gaussian, Log-Normal, and Exponential models via Kolmogorov-Smirnov goodness-of-fit evaluations.
3. **Hypothesis Testing & Collinearity Diagnostics (CO4):** Validate statistical differences between buyers and non-buyers using non-parametric Mann-Whitney U tests, Chi-Square tests of independence, and Variance Inflation Factor (VIF) collinearity audits.
4. **Develop Deep Learning Architecture (CO5):** Formulate and train a PyTorch Deep Neural Network (ShoppersMLP) tailored for tabular session data under an imbalanced 5.46:1 non-purchaser to purchaser ratio.
5. **Rigorous Benchmarking & Business Translation (CO5):** Benchmark against industry-standard machine learning baselines (Logistic Regression, Random Forest), translate analytical insights into actionable e-commerce strategies, and deploy an interactive Streamlit web dashboard.

---

## 3. Dataset Description, Source, and Statistical Overview

### 3.1 Data Source and Provenance
- **Dataset:** Online Shoppers Purchasing Intention Dataset (`online_shoppers_intention.csv`)
- **Source:** UC Irvine Machine Learning Repository
- **Direct UCI URL:** [https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **DOI:** [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)
- **Authors:** C. Okan Sakar and Yomi Kastro (Bogazici University & Inveon)
- **Reference Paper:** C. O. Sakar, S. O. Polat, M. Katircioglu, and Y. Kastro. *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications, 31(10):6893-6908, October 2019.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

### 3.2 Dataset Dimensions and Schema
The dataset records **12,330 distinct user browsing sessions** over a 1-year calendar window across **18 attributes** (10 continuous numerical features, 7 discrete/categorical features, and 1 binary classification target `Revenue`).

#### Table 1: Comprehensive Data Dictionary
| # | Attribute | Type | Scale / Range | Semantic Description |
|---|---|---|---|---|
| 1 | `Administrative` | Integer | 0 – 27 | Number of account/management pages visited in the session |
| 2 | `Administrative_Duration` | Float | 0.0 – 3,398.8 s | Time spent on administrative pages (seconds) |
| 3 | `Informational` | Integer | 0 – 24 | Number of informational/help pages visited |
| 4 | `Informational_Duration` | Float | 0.0 – 2,549.4 s | Time spent on informational pages (seconds) |
| 5 | `ProductRelated` | Integer | 0 – 705 | Number of product-related catalog pages visited |
| 6 | `ProductRelated_Duration` | Float | 0.0 – 63,973.5 s | Total duration spent viewing product pages (seconds) |
| 7 | `BounceRates` | Float | 0.0 – 0.2 | Percentage of visitors who enter the page and exit without further action |
| 8 | `ExitRates` | Float | 0.0 – 0.2 | Percentage of pageviews that were the final view in the session |
| 9 | `PageValues` | Float | 0.0 – 361.8 | Google Analytics average value of pages viewed prior to transaction |
| 10 | `SpecialDay` | Float | 0.0 – 1.0 | Closeness of browsing date to a holiday/festival (e.g., Valentine's Day) |
| 11 | `Month` | Categorical | Feb – Dec (10) | Calendar month of session occurrence |
| 12 | `OperatingSystems` | Discrete | 1 – 8 | Operating system identifier used by visitor |
| 13 | `Browser` | Discrete | 1 – 13 | Web browser identifier used by visitor |
| 14 | `Region` | Discrete | 1 – 9 | Geographic region code of visitor |
| 15 | `TrafficType` | Discrete | 1 – 20 | Web traffic acquisition source type (direct, search, referral, ads) |
| 16 | `VisitorType` | Categorical | 3 classes | `Returning_Visitor`, `New_Visitor`, `Other` |
| 17 | `Weekend` | Boolean | `True`, `False` | Boolean indicator whether session occurred on a weekend |
| 18 | `Revenue` | Boolean Target | `True`, `False` | Target: Did the browsing session culminate in a purchase? |

### 3.3 Class Imbalance Profile
- Non-Purchasing Sessions (`Revenue = False`): **10,422 sessions (84.53%)**
- Purchasing Sessions (`Revenue = True`): **1,908 sessions (15.47%)**
- Imbalance Ratio: **5.46 : 1**.

---

## 4. Preprocessing and Data Hygiene Pipeline

### 4.1 Duplicate Detection & Removal
Inspection identified **125 exact duplicate session rows**, which were pruned to preserve independent statistical observations, leaving 12,205 unique sessions.

### 4.2 Missingness Analysis
Audit confirmed **0 missing or null entries** across all 18 columns. The dataset is fully complete.

### 4.3 Outlier Analysis & Winsorization Capping
Tukey's Interquartile Range (IQR) method detected heavy right tails in `ProductRelated_Duration`, `PageValues`, and `Administrative_Duration`. In e-commerce, power shoppers who browse extensively or generate high PageValues represent high-propensity buyers. Row-wise deletion would eliminate the highest-value transactions. Consequently, non-destructive **IQR Capping (Winsorization)** was applied at $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$.

### 4.4 Leakage-Free Stratified Partitioning & Scaling
To ensure **zero data leakage**:
1. The dataset was split into **Train (70%, 8,543 sessions)**, **Validation (15%, 1,831 sessions)**, and **Held-Out Test (15%, 1,831 sessions)** subsets using stratified sampling on `Revenue`.
2. Categorical and discrete identifiers (`Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`) were one-hot encoded, expanding input dimensionality to **74 features**.
3. A `RobustScaler` was fitted **strictly on the 8,543 training sessions**, and subsequently transformed across validation and test sets.

---

## 5. Exploratory Data Analysis & Visualizations (CO3, CO4)

### Visualization 1: Univariate Continuous Distributions (KDE Overlays)
- **Chart Type:** Multi-panel Histograms with Kernel Density Estimation (KDE) and Mean/Median reference lines.
- **Reason for Selection:** Continuous e-commerce metrics exhibit extreme positive skewness requiring empirical density visualization.
- **Observed Pattern:** `PageValues` is centered at a median of 0.00 ($\mu = 5.89, \sigma = 18.57$), as the vast majority of non-converting sessions register no page value. `ProductRelated_Duration` is right-skewed with a median of 598.9 seconds (~10 minutes). `ExitRates` and `BounceRates` display concentrated peaks near 0.01–0.03.
- **Supported Insight:** Standard Gaussian assumptions fail for web telemetry; non-parametric and robust scaling treatments are essential.

### Visualization 2: Outlier Boxplot Inspection
- **Chart Type:** Multi-panel Tukey Boxplots across numerical session features.
- **Reason for Selection:** Identifies distribution quartiles, medians, and whisker boundaries ($1.5 \times IQR$).
- **Observed Pattern:** Severe upper-tail outliers exist in `ProductRelated` catalog views (up to 705 pages) and `Administrative_Duration` (up to 3,398 seconds).
- **Supported Insight:** Capping upper bounds preserves these highly active converting sessions while preventing gradient instability during neural network backpropagation.

### Visualization 3: Pearson Correlation Heatmap
- **Chart Type:** Triangular Correlation Heatmap with Diverging Palette and Value Annotations.
- **Reason for Selection:** Evaluates pairwise linear relationships and diagnoses collinearity among web metrics.
- **Observed Pattern:** Extreme positive correlation exists between `BounceRates` and `ExitRates` ($r = 0.91$), and between `ProductRelated` and `ProductRelated_Duration` ($r = 0.86$). Conversely, `PageValues` exhibits moderate positive correlation with transaction completion ($r = 0.49$).
- **Supported Insight:** Bouncing and exiting are strongly coupled; high bounce pages directly elevate session abandonment.

### Visualization 4: Bivariate Relationship (PageValues & ExitRates vs Revenue)
- **Chart Type:** Boxplot of PageValues by Purchase Outcome and Density Plot of ExitRates.
- **Reason for Selection:** Directly contrasts behavioral metrics between purchasing and non-purchasing visitors.
- **Observed Pattern:** Non-purchasing sessions exhibit a median `PageValues` of **0.00** ($\mu = 1.98$). In contrast, purchasing sessions exhibit a mean `PageValues` of **27.26** ($\text{median} = 16.76$). Density of ExitRates reveals purchasers cluster tightly below 0.02, while non-purchasers span out past 0.10.
- **Supported Insight:** Mann-Whitney U test confirms that `PageValues` is the single most predictive feature ($U = 3,421,902.5, p < 10^{-15}, \text{Cohen's } d = 1.15$).

### Visualization 5: Monthly Conversion Rates vs Total Traffic
- **Chart Type:** Dual-Axis Bar Chart (Conversion Rate %) and Line Chart (Total Session Count).
- **Reason for Selection:** Disentangles traffic volume from actual purchasing intent across calendar seasonality.
- **Observed Pattern:** May generates the highest raw traffic volume (~3,364 sessions) but a modest conversion rate (~10.9%). November (Black Friday / Cyber Week) generates massive traffic (~2,998 sessions) and a peak conversion rate of **25.4%**.
- **Supported Insight:** Marketing campaign budgets should be heavily weighted towards Q4 holiday shopping periods where conversion yields double.

### Visualization 6: Multivariate Interaction (ExitRates vs PageValues by Outcome)
- **Chart Type:** Two-Dimensional Scatter Plot colored by Transaction Outcome (`Revenue`).
- **Reason for Selection:** Illustrates the multivariate decision boundary between high-value engagement and session abandonment.
- **Observed Pattern:** Purchasing sessions (`Revenue = True`) occupy the high-PageValues, low-ExitRates quadrant. Non-purchasing sessions dominate the low-PageValues, high-ExitRates region.
- **Supported Insight:** Visitors navigating through high-value pages with low friction are overwhelmingly likely to purchase.

### Visualization 7: Probability Distribution Fit & Q-Q Plot Analysis
- **Chart Type:** Empirical Histogram with Fitted Gaussian PDF alongside Standard Normal Q-Q Plot.
- **Reason for Selection:** Formally tests whether `ProductRelated_Duration` adheres to a normal distribution.
- **Observed Pattern:** The empirical distribution deviates sharply from the 45-degree normal reference line with massive right-tail departure ($Z > 2.0$).
- **Supported Insight:** Kolmogorov-Smirnov test firmly rejects normality ($p < 10^{-15}$), proving that linear Gaussian models are inappropriate and justifying non-linear deep neural networks.

---

## 6. Statistical Hypothesis Testing & Multicollinearity Diagnostics (CO4)

### 6.1 Non-Parametric Mann-Whitney U Tests (Numeric vs Revenue)
Given acute skewness across session features, Mann-Whitney U tests were conducted:
- `PageValues`: $p = 0.0000$, Cohen's $d = 1.148$ (Very large positive effect size)
- `ExitRates`: $p = 0.0000$, Cohen's $d = -0.638$ (Moderate negative effect size)
- `ProductRelated_Duration`: $p = 0.0000$, Cohen's $d = 0.412$ (Moderate positive effect size)
- `BounceRates`: $p = 0.0000$, Cohen's $d = -0.528$ (Moderate negative effect size)

### 6.2 Pearson Chi-Square Tests of Independence (Categorical vs Revenue)
Contingency tables evaluated categorical dependence:
- `Month`: $\chi^2 = 383.5, \text{dof} = 9, p < 10^{-15}, \text{Cramer's } V = 0.177$
- `VisitorType`: $\chi^2 = 135.2, \text{dof} = 2, p < 10^{-15}, \text{Cramer's } V = 0.105$ (New visitors convert at 24.9% vs 14.6% for returning visitors)
- `TrafficType`: $\chi^2 = 275.1, \text{dof} = 19, p < 10^{-15}, \text{Cramer's } V = 0.150$

### 6.3 Multicollinearity Diagnostics via Variance Inflation Factor (VIF)
- `ExitRates`: $\text{VIF} = 10.4$ (High Multicollinearity, driven by BounceRates coupling)
- `BounceRates`: $\text{VIF} = 9.8$ (High Multicollinearity)
- `ProductRelated`: $\text{VIF} = 4.2$ (Moderate Multicollinearity)
- `PageValues`: $\text{VIF} = 1.08$ (Low Multicollinearity, orthogonal predictive signal)

---

## 7. Deep Learning Model Architecture & Training (CO5)

### 7.1 Architecture Design: PyTorch ShoppersMLP
To capture non-linear interactions across the 74 engineered features, we developed a deep tabular neural network:
- **Input Layer:** 74 engineered features
- **Layer 1:** Fully Connected `Linear(74, 128)` $\rightarrow$ `BatchNorm1d(128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.30)`
- **Layer 2:** Fully Connected `Linear(128, 64)` $\rightarrow$ `BatchNorm1d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.21)`
- **Layer 3:** Fully Connected `Linear(64, 32)` $\rightarrow$ `BatchNorm1d(32)` $\rightarrow$ `ReLU`
- **Output Layer:** Fully Connected `Linear(32, 1)` (produces raw logit output for numerical stability with BCE)

### 7.2 Loss Function & Class Imbalance Handling
Standard Cross-Entropy leads to majority-class bias on imbalanced data. We deployed **Weighted Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`)**, setting the positive class weight:
$$\text{pos\_weight} = \frac{N_{\text{non-purchasers}}}{N_{\text{purchasers}}} = \frac{7,295}{1,248} \approx 5.85$$
This penalizes False Negatives ~6x, compelling the optimizer to detect true purchasing sessions.

### 7.3 Optimization & Regularization Strategy
- **Optimizer:** `AdamW` ($\text{lr} = 0.001$, weight decay $= 10^{-4}$).
- **Learning Rate Scheduler:** `ReduceLROnPlateau(factor=0.5, patience=2, mode='min')`.
- **Batch Size:** 128 samples per mini-batch.
- **Early Stopping:** Monitored on validation loss with patience of 6 epochs. Training reached its optimal validation checkpoint at **Epoch 5 ($\text{Val Loss} = 0.9797$)**.

---

## 8. Model Evaluation Visualizations & Benchmarking (CO5)

### 8.1 Quantitative Benchmark on Held-Out Test Set (1,831 Sessions)
The PyTorch ShoppersMLP was evaluated against class-weighted Logistic Regression and class-weighted Random Forest (100 estimators).

#### Table 2: Test Set Benchmark Results
| Evaluation Metric | Logistic Regression (Baseline) | Random Forest (Baseline) | PyTorch ShoppersMLP (Deep Learning) |
|---|---|---|---|
| **Test Accuracy** | 64.50% | **68.87%** | 64.99% |
| **Recall (Sensitivity)** | 73.78% | 69.23% | **75.17%** |
| **Precision** | 26.84% | **29.12%** | 27.39% |
| **F1-Score** | 0.3937 | **0.4099** | 0.4015 |
| **Balanced Accuracy** | 68.28% | 69.02% | **69.14%** |
| **Specificity** | 62.78% | **68.80%** | 63.11% |
| **ROC-AUC** | 0.7462 | **0.7755** | 0.7722 |
| **PR-AUC (Average Precision)** | 0.3255 | **0.3762** | 0.3647 |

### 8.2 Evaluation Charts Analysis
1. **Loss Curves (Figure 8):** Training loss decreases steadily from 1.0614 to 0.8636; validation loss stabilizes around 0.9797, with early stopping halting training before overfitting.
2. **Accuracy Curves (Figure 9):** Training accuracy climbs from 50.5% to 69.7%, with validation accuracy tracking near 63.0%.
3. **Confusion Matrix Heatmap (Figure 10):** Evaluated at threshold 0.5, ShoppersMLP identifies **215 True Positives out of 286 actual purchasing sessions (75.17% Recall)**, incurring only 71 False Negatives.
4. **ROC Curve (Figure 11):** Achieves an **AUC of 0.7722**, well above the diagonal baseline (0.5000), proving strong discriminative capacity.
5. **Precision-Recall Curve (Figure 12):** Achieves an **Average Precision of 0.3647**, more than double the random prevalence baseline of 0.1547.
6. **Benchmark Bar Chart (Figure 13):** Shows ShoppersMLP achieving the highest Recall (75.17%), while Random Forest delivers higher overall Precision and F1-score.

---

## 9. Key Findings & Strategic Recommendations

1. **PageValues is the Decisive Behavioral Trigger:** Sessions where `PageValues > 0` exhibit conversion rates of **54.2%**, compared to <4% for sessions with zero PageValues. Marketing teams should direct ad spend towards landing pages that funnel users into high-PageValue product paths.
2. **Friction Reduction on High-Exit Pages:** Product catalog pages exhibiting exit rates $> 0.05$ require immediate UX redesign (e.g. simplified variant selection, sticky checkout buttons).
3. **Capitalize on Seasonal Surges:** November conversion rates (25.4%) double average monthly yields. E-commerce platforms should scale up server capacity and promotional campaigns during Q4.
4. **Deploy Real-Time Cart Abandonment Popups:** Using ShoppersMLP to flag wavering shoppers (predicted purchase probability 35%–50%) enables real-time dynamic discounting to nudge them to checkout.

---

## 10. Challenges and Limitations

1. **Session-Level Aggregation:** The dataset provides cumulative session totals rather than clickstream timestamps. Fine-grained sequence modeling with LSTMs or Transformers requires time-stamped event logs.
2. **Precision-Recall Trade-Off:** High recall (75.17%) comes with False Positives (570 non-purchasers predicted as buyers). In practice, thresholds can be tuned based on incentive cost.

---

## 11. Conclusion and Future Scope

### 11.1 Conclusion
This study developed an end-to-end exploratory analysis and deep learning architecture for e-commerce purchasing intention prediction fulfilling Course Outcomes CO3, CO4, and CO5:
- Investigated 12,330 user browsing sessions across 18 multi-modal features.
- Quantified distribution non-normality and statistical feature dependencies.
- Built, trained, and benchmarked a PyTorch ShoppersMLP achieving 75.17% recall and 0.7722 ROC-AUC.
- Implemented an interactive Streamlit web dashboard.

### 11.2 Future Scope
1. **Sequential Clickstream Modeling:** Implement Transformer-based attention models on raw URL navigation sequences.
2. **Explainable AI (SHAP):** Integrate per-session SHAP feature contributions for real-time customer journey analytics.

---

## 12. References (IEEE Style)
1. C. O. Sakar, S. O. Polat, M. Katircioglu, and Y. Kastro, "Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks," *Neural Computing and Applications*, vol. 31, no. 10, pp. 6893–6908, Oct. 2019. DOI: 10.1007/s00521-018-3523-0.
2. UC Irvine Machine Learning Repository, "Online Shoppers Purchasing Intention Dataset," DOI: 10.24432/C5F88Q, 2019. [Online]. Available: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset.
3. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS 32)*, 2019, pp. 8024–8035.
4. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research (JMLR)*, vol. 12, pp. 2825–2830, 2011.
5. J. Tukey, *Exploratory Data Analysis*, Reading, MA: Addison-Wesley, 1977.

---

## 13. Appendix: Source Code Structure & Execution Instructions
The complete project repository is modularly arranged:
- Data Loading: `src/data_loader.py`
- Preprocessing & Leakage-Free Scaling: `src/preprocessing.py`
- EDA & Statistical Tests: `src/eda.py`
- Visualizations: `src/visualizations.py`
- Deep Learning Architecture: `src/model.py`
- Model Evaluation: `src/evaluation.py`
- Streamlit Dashboard: `app.py`
- Automated Test Suite: `tests/test_pipeline.py`
- Master Pipeline Runner: `run_pipeline.py`
