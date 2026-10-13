"""
Academic PDF Report Generator for Online Shoppers Purchasing Intention Project
Generates docs/project_report.pdf using ReportLab with clean styling,
academic header/footer, structured tables, and embedded high-res figures.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Custom canvas that adds dynamic 'Page X of Y' footers and top headers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))

        # Header
        self.drawString(54, 750, "KPRIET | Dept. of AI & DS — Exploratory Data Analysis & Visualization (Assignment II)")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 744, 558, 744)

        # Footer
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Case Study: Online Shoppers Purchasing Intention EDA & Deep Learning")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def generate_pdf_report(output_pdf: str = "docs/project_report.pdf"):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=1,
        spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#3B82F6"),
        alignment=1,
        spaceAfter=25
    )
    meta_box_style = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1F2937")
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#374151"),
        spaceAfter=7
    )
    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#6B7280"),
        alignment=1,
        spaceAfter=10
    )

    story = []

    # ==========================================
    # TITLE COVER PAGE
    # ==========================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("KPR INSTITUTE OF ENGINEERING AND TECHNOLOGY", ParagraphStyle("Inst", fontName="Helvetica-Bold", fontSize=13, alignment=1, textColor=colors.HexColor("#1F2937"))))
    story.append(Paragraph("DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE", ParagraphStyle("Dept", fontName="Helvetica", fontSize=10, alignment=1, textColor=colors.HexColor("#4B5563"))))
    story.append(Paragraph("Coimbatore – 641 407, Tamil Nadu", ParagraphStyle("Loc", fontName="Helvetica", fontSize=9, alignment=1, textColor=colors.HexColor("#6B7280"))))
    story.append(Spacer(1, 40))

    story.append(Paragraph("CASE STUDY: EXPLORATORY DATA ANALYSIS AND VISUALIZATION OF A REAL-WORLD DATASET", title_style))
    story.append(Paragraph("Online Shoppers Purchasing Intention Prediction via Statistical EDA & PyTorch Deep Learning", subtitle_style))
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#2563EB"), spaceAfter=35))

    meta_table_data = [
        [Paragraph("<b>Course:</b>", meta_box_style), Paragraph("Exploratory Data Analysis and Visualization (21AD401)", meta_box_style)],
        [Paragraph("<b>Academic Year:</b>", meta_box_style), Paragraph("2026–2027 (Semester VI)", meta_box_style)],
        [Paragraph("<b>Assignment:</b>", meta_box_style), Paragraph("Assignment II (Total Marks: 40)", meta_box_style)],
        [Paragraph("<b>Course Outcomes:</b>", meta_box_style), Paragraph("CO3 (Visualization), CO4 (EDA & Testing), CO5 (Deep Learning)", meta_box_style)],
        [Paragraph("<b>Dataset Source:</b>", meta_box_style), Paragraph("UC Irvine Machine Learning Repository (DOI: 10.24432/C5F88Q)", meta_box_style)],
        [Paragraph("<b>Student Name:</b>", meta_box_style), Paragraph("[Student Name Placeholder]", meta_box_style)],
        [Paragraph("<b>Register Number:</b>", meta_box_style), Paragraph("[Register Number Placeholder]", meta_box_style)],
        [Paragraph("<b>Date of Submission:</b>", meta_box_style), Paragraph("October 2026", meta_box_style)]
    ]
    t_meta = Table(meta_table_data, colWidths=[130, 340])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_meta)
    story.append(PageBreak())

    # ==========================================
    # SECTION 1: ABSTRACT
    # ==========================================
    story.append(Paragraph("1. Abstract", h1_style))
    story.append(Paragraph(
        "In modern electronic commerce, understanding visitor browsing dynamics is critical for conversion rate optimization and personalized shopping experiences. However, the vast majority of e-commerce user sessions (~84.5%) terminate without completing a purchase transaction. This case study conducts an exhaustive exploratory analysis and predictive modeling workflow on the genuine UC Irvine Online Shoppers Purchasing Intention benchmark dataset, comprising 12,330 multi-session records across 18 behavioral and technical attributes. We implement an automated preprocessing pipeline that resolves session duplicates, applies Tukey IQR Winsorization capping to mitigate extreme variance while retaining high-value purchasing sessions, and guarantees zero data leakage through strictly partitioned feature scaling. Non-parametric Mann-Whitney U tests (p &lt; 10<sup>-15</sup>) identify PageValues and ExitRates as primary determinants of purchase propensity. We design, optimize, and train a PyTorch Multilayer Perceptron (ShoppersMLP) with Batch Normalization, Dropout regularization, and class-weighted Binary Cross-Entropy loss (pos_weight &approx; 5.46). Evaluated on a held-out test set of 1,831 sessions, ShoppersMLP attains a recall of 75.17% and an ROC-AUC of 0.7722, accurately capturing purchasing intent and offering clear trade-offs against Random Forest and Logistic Regression baselines.",
        body_style
    ))

    # ==========================================
    # SECTION 2: INTRODUCTION & OBJECTIVES
    # ==========================================
    story.append(Paragraph("2. Introduction & Objectives", h1_style))
    story.append(Paragraph(
        "Electronic retail platforms process massive volumes of browsing telemetry every second. The conversion rate—the fraction of browsing sessions culminating in a monetary purchase—typically hovers around 2% to 15% across global online retail. Traditional batch analytics review customer transactions post-hoc, precluding timely interventions. Developing predictive models capable of real-time purchasing intention prediction from ongoing session telemetry empowers e-commerce systems to trigger dynamic micro-incentives (e.g. customized discounts, free shipping vouchers, or chatbot assistance) to convert wavering visitors into buyers.",
        body_style
    ))

    # ==========================================
    # SECTION 3: DATASET DESCRIPTION & SCHEMA
    # ==========================================
    story.append(Paragraph("3. Dataset Description & Provenance", h1_style))
    story.append(Paragraph(
        "<b>Source:</b> UC Irvine Machine Learning Repository (DOI: 10.24432/C5F88Q, Sakar et al., 2019).<br/>"
        "<b>Dimensions:</b> 12,330 user browsing sessions across 18 attributes (10 continuous numerical features, 7 discrete/categorical features, 1 binary target Revenue).<br/>"
        "<b>Class Distribution:</b> 10,422 non-purchasing sessions (False, 84.53%) vs 1,908 purchasing sessions (True, 15.47%), exhibiting an imbalanced ratio of ~5.46:1.",
        body_style
    ))

    # Data Dictionary Table Summary
    dict_table_data = [
        ["#", "Attribute", "Data Type", "Role", "Description"],
        ["1", "Administrative", "Numeric", "Feature", "Admin pageviews in session (0-27)"],
        ["2", "Administrative_Duration", "Numeric", "Feature", "Time on admin pages in seconds"],
        ["3", "Informational", "Numeric", "Feature", "Informational pageviews (0-24)"],
        ["4", "Informational_Duration", "Numeric", "Feature", "Time on info pages in seconds"],
        ["5", "ProductRelated", "Numeric", "Feature", "Product catalog pageviews (0-705)"],
        ["6", "ProductRelated_Duration", "Numeric", "Feature", "Time on product pages in seconds"],
        ["7", "BounceRates", "Numeric", "Feature", "Average bounce rate of visited pages"],
        ["8", "ExitRates", "Numeric", "Feature", "Average exit rate of visited pages"],
        ["9", "PageValues", "Numeric", "Feature", "Google Analytics average page value"],
        ["10", "SpecialDay", "Numeric", "Feature", "Closeness to holiday/festival (0 to 1)"],
        ["11", "Month", "Categorical", "Feature", "Calendar month of session (Feb-Dec)"],
        ["12", "VisitorType", "Categorical", "Feature", "Visitor classification (Returning/New/Other)"],
        ["13", "Weekend", "Boolean", "Feature", "Session occurred on weekend (True/False)"],
        ["14", "Revenue", "Boolean Target", "Target", "Transaction completed? (True/False)"]
    ]
    t_dict = Table(dict_table_data, colWidths=[20, 110, 65, 60, 215])
    t_dict.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")])
    ]))
    story.append(t_dict)
    story.append(Paragraph("Table 1: Selected Key Attributes from Data Dictionary (all 18 features used in modeling).", caption_style))

    # ==========================================
    # SECTION 4: PREPROCESSING & DATA HYGIENE
    # ==========================================
    story.append(Paragraph("4. Preprocessing & Leakage-Free Pipeline", h1_style))
    story.append(Paragraph(
        "<b>Deduplication:</b> 125 duplicate session records were pruned. Zero missing values exist across all attributes.<br/>"
        "<b>Outlier Treatment:</b> Tukey's IQR detection revealed extreme tails in ProductRelated_Duration and PageValues. To avoid deleting high-value purchasing customers, IQR capping (Winsorization) was applied.<br/>"
        "<b>Leakage-Free Partitioning & Scaling:</b> Stratified split into Train (70%, 8,543), Validation (15%, 1,831), and Held-Out Test (15%, 1,831) subsets. Categorical features were one-hot encoded to 74 dimensions. A RobustScaler was fitted strictly on the training partition and transformed across validation/test partitions.",
        body_style
    ))

    # ==========================================
    # SECTION 5: EXPLORATORY VISUALIZATIONS (CO3)
    # ==========================================
    story.append(PageBreak())
    story.append(Paragraph("5. Exploratory Data Analysis & Visualizations (CO3, CO4)", h1_style))

    if os.path.exists("outputs/figures/eda_1_univariate_distributions.png"):
        story.append(Image("outputs/figures/eda_1_univariate_distributions.png", width=470, height=260))
        story.append(Paragraph("Figure 1: Univariate distributions of PageValues, ProductDuration, ExitRates, and BounceRates with KDE overlays.", caption_style))
        story.append(Paragraph("<b>Insight:</b> PageValues is heavily right-skewed with median 0.00. ExitRates and BounceRates peak sharply between 0.01 and 0.03.", body_style))

    if os.path.exists("outputs/figures/eda_3_correlation_heatmap.png"):
        story.append(Spacer(1, 10))
        story.append(Image("outputs/figures/eda_3_correlation_heatmap.png", width=380, height=270))
        story.append(Paragraph("Figure 2: Pearson Correlation Heatmap across continuous web browsing features.", caption_style))
        story.append(Paragraph("<b>Insight:</b> Strong multicollinearity exists between BounceRates and ExitRates (r = 0.91), and ProductRelated and Duration (r = 0.86).", body_style))

    story.append(PageBreak())
    if os.path.exists("outputs/figures/eda_4_duration_vs_deposit.png"):
        story.append(Image("outputs/figures/eda_4_duration_vs_deposit.png", width=470, height=210))
        story.append(Paragraph("Figure 3: Bivariate analysis of PageValues and ExitRates vs transaction outcome (Revenue).", caption_style))
        story.append(Paragraph("<b>Insight:</b> Purchasing sessions exhibit a mean PageValues of 27.26 vs 1.98 for non-purchasers (Mann-Whitney U p &lt; 10<sup>-15</sup>).", body_style))

    if os.path.exists("outputs/figures/eda_5_job_subscription_rates.png"):
        story.append(Spacer(1, 10))
        story.append(Image("outputs/figures/eda_5_job_subscription_rates.png", width=470, height=210))
        story.append(Paragraph("Figure 4: Conversion rates vs total session traffic across calendar months.", caption_style))
        story.append(Paragraph("<b>Insight:</b> November generates peak conversion yield (25.4%), doubling the average conversion rate of spring and summer months.", body_style))

    # ==========================================
    # SECTION 6: STATISTICAL TESTING & DISTRIBUTIONS
    # ==========================================
    story.append(Paragraph("6. Statistical Hypothesis Testing & Distributions (CO4)", h1_style))
    story.append(Paragraph(
        "<b>Goodness-of-Fit Testing:</b> Kolmogorov-Smirnov and D'Agostino-Pearson tests firmly rejected Gaussian normality for PageValues and ProductRelated_Duration (p &lt; 10<sup>-15</sup>). Empirical metrics display extreme positive skewness.<br/>"
        "<b>Mann-Whitney U Tests:</b> Confirmed that PageValues (p=0.0000, Cohen's d=1.148) and ExitRates (p=0.0000, Cohen's d=-0.638) are statistically significant discriminators.<br/>"
        "<b>Chi-Square Tests of Independence:</b> Month (Chi2=383.5, p=0.0000, Cramer's V=0.177) and VisitorType (Chi2=135.2, p=0.0000) show strong dependence on purchase completion.",
        body_style
    ))

    # ==========================================
    # SECTION 7 & 8: DEEP LEARNING MODEL & EVALUATION (CO5)
    # ==========================================
    story.append(PageBreak())
    story.append(Paragraph("7. Deep Learning Model & Evaluation (CO5)", h1_style))
    story.append(Paragraph(
        "<b>PyTorch ShoppersMLP Architecture:</b> Input(74) &rarr; Linear(128) &rarr; BatchNorm1d &rarr; ReLU &rarr; Dropout(0.3) &rarr; Linear(64) &rarr; BatchNorm1d &rarr; ReLU &rarr; Dropout(0.2) &rarr; Linear(32) &rarr; ReLU &rarr; Linear(1).<br/>"
        "<b>Optimization & Regularization:</b> Trained using AdamW (lr=0.001, weight decay=1e-4) with pos_weight=5.85 in BCEWithLogitsLoss to handle the 5.46:1 class imbalance. Early stopping concluded training at Epoch 5 with optimal validation loss 0.9797.",
        body_style
    ))

    # Benchmark Table
    bench_data = [
        ["Model Architecture", "Accuracy", "Recall", "Precision", "F1-Score", "ROC-AUC", "PR-AUC"],
        ["Logistic Regression (Baseline)", "64.50%", "73.78%", "26.84%", "0.3937", "0.7462", "0.3255"],
        ["Random Forest (Baseline)", "68.87%", "69.23%", "29.12%", "0.4099", "0.7755", "0.3762"],
        ["PyTorch ShoppersMLP (Deep Learning)", "64.99%", "75.17%", "27.39%", "0.4015", "0.7722", "0.3647"]
    ]
    t_bench = Table(bench_data, colWidths=[150, 50, 50, 50, 50, 50, 50])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")])
    ]))
    story.append(t_bench)
    story.append(Paragraph("Table 2: Performance metrics on held-out test set (1,831 browsing sessions).", caption_style))

    # Figures: Loss Curves & Confusion Matrix
    if os.path.exists("outputs/figures/model_1_loss_curves.png") and os.path.exists("outputs/figures/model_3_confusion_matrix.png"):
        story.append(Spacer(1, 10))
        t_img = Table([
            [
                Image("outputs/figures/model_1_loss_curves.png", width=230, height=155),
                Image("outputs/figures/model_3_confusion_matrix.png", width=230, height=155)
            ]
        ], colWidths=[240, 240])
        t_img.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(t_img)
        story.append(Paragraph("Figure 5: (Left) Loss convergence across epochs; (Right) Test Confusion Matrix (75.17% Recall).", caption_style))

    if os.path.exists("outputs/figures/model_4_roc_auc_curve.png") and os.path.exists("outputs/figures/model_6_baseline_vs_mlp_comparison.png"):
        story.append(Spacer(1, 5))
        t_img2 = Table([
            [
                Image("outputs/figures/model_4_roc_auc_curve.png", width=230, height=155),
                Image("outputs/figures/model_6_baseline_vs_mlp_comparison.png", width=230, height=155)
            ]
        ], colWidths=[240, 240])
        t_img2.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(t_img2)
        story.append(Paragraph("Figure 6: (Left) ROC Curve (AUC = 0.7722); (Right) Benchmark comparison against baseline models.", caption_style))

    # ==========================================
    # SECTION 9: FINDINGS, CONCLUSION & REFERENCES
    # ==========================================
    story.append(PageBreak())
    story.append(Paragraph("8. Key Findings & Strategic Recommendations", h1_style))
    story.append(Paragraph(
        "1. <b>PageValues Dominance:</b> Sessions with PageValues &gt; 0 achieve a conversion rate of 54.2%, compared to &lt;4% for zero-value sessions. Paid advertising should funnel traffic directly into high-value product paths.<br/>"
        "2. <b>Friction Reduction:</b> Product catalog pages exhibiting exit rates &gt; 0.05 require UX restructuring to streamline add-to-cart actions.<br/>"
        "3. <b>Seasonal Campaign Timing:</b> November conversion yield (25.4%) doubles the rest of the year. Marketing budgets should be heavily weighted towards Q4 holiday periods.<br/>"
        "4. <b>Real-Time Dynamic Incentives:</b> Deploying ShoppersMLP to flag wavering shoppers (predicted purchase probability 35%–50%) enables real-time dynamic discounting to nudge them to checkout.",
        body_style
    ))

    story.append(Paragraph("9. Conclusion", h1_style))
    story.append(Paragraph(
        "This case study successfully implemented an end-to-end data science and deep learning workflow on 12,330 e-commerce user sessions fulfilling all course outcomes (CO3, CO4, CO5). We formulated a leakage-free preprocessing pipeline, performed rigorous hypothesis testing, trained an effective PyTorch ShoppersMLP classifier, and deployed an interactive Streamlit application.",
        body_style
    ))

    story.append(Paragraph("10. References (IEEE Style)", h1_style))
    refs = [
        "[1] C. O. Sakar, S. O. Polat, M. Katircioglu, and Y. Kastro, 'Real-time prediction of online shoppers\' purchasing intention using multilayer perceptron and LSTM recurrent neural networks,' <i>Neural Computing and Applications</i>, vol. 31, no. 10, pp. 6893–6908, Oct. 2019. DOI: 10.1007/s00521-018-3523-0.",
        "[2] UC Irvine Machine Learning Repository, 'Online Shoppers Purchasing Intention Dataset,' DOI: 10.24432/C5F88Q, 2019. Available: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset.",
        "[3] A. Paszke et al., 'PyTorch: An imperative style, high-performance deep learning library,' in <i>Advances in Neural Information Processing Systems (NeurIPS 32)</i>, 2019, pp. 8024–8035.",
        "[4] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python,' <i>Journal of Machine Learning Research (JMLR)</i>, vol. 12, pp. 2825–2830, 2011.",
        "[5] J. Tukey, <i>Exploratory Data Analysis</i>, Reading, MA: Addison-Wesley, 1977."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle("Ref", fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor("#4B5563"), spaceAfter=5)))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated academic PDF report: {output_pdf}")


if __name__ == "__main__":
    generate_pdf_report()
