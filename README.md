# Online Shoppers Purchasing Intention Dataset Documentation

## 1. Overview & Provenance
- **Dataset Name:** Online Shoppers Purchasing Intention Dataset (`online_shoppers_intention.csv`)
- **Source:** UC Irvine Machine Learning Repository
- **Direct UCI URL:** [https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **DOI:** [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)
- **Original Authors:** C. Okan Sakar and Yomi Kastro (Bogazici University & Inveon Information Technologies)
- **Reference Paper:**
  > C. O. Sakar, S. O. Polat, M. Katircioglu, and Y. Kastro. *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications, 31(10):6893-6908, October 2019.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 2. Business Context & Objective
The dataset consists of feature vectors belonging to **12,330 distinct online user browsing sessions** over a 1-year calendar window on a major e-commerce retail platform. Each session was formed so that each record corresponds to a different user over a 1-year period to avoid any bias to a specific campaign, special day, user profile, or period.

The primary machine learning objective is binary classification: predict whether an online browsing session will finalize in a financial transaction / purchase (`Revenue = True`), enabling e-commerce platforms to:
1. Dynamically target hesitant visitors with real-time conversion incentives (discounts, chat assistance).
2. Optimize navigation architecture by detecting pages with elevated exit and bounce rates.
3. Maximize overall gross merchandise value (GMV) and return on ad spend (ROAS).

---

## 3. Dataset Characteristics
- **Total Records (Sessions):** 12,330
- **Total Features:** 17 predictor attributes + 1 binary target = **18 features**
- **Data Types:** 10 Numerical features, 7 Categorical / Discrete attributes, 1 Binary Target
- **Target Distribution:**
  - `False` (Did not purchase): 10,422 sessions (84.53%)
  - `True` (Completed purchase): 1,908 sessions (15.47%)
  - Realistic class imbalance ratio (~5.46 : 1).

---

## 4. Comprehensive Data Dictionary

| # | Feature Name | Data Type | Value Range / Classes | Description |
|---|---|---|---|---|
| 1 | `Administrative` | Integer | 0 to 27 | Number of administrative pages visited by the user in this session |
| 2 | `Administrative_Duration` | Float | 0.0 to 3,398.75 | Total time spent on administrative pages (in seconds) |
| 3 | `Informational` | Integer | 0 to 24 | Number of informational pages visited by the user |
| 4 | `Informational_Duration` | Float | 0.0 to 2,549.38 | Total time spent on informational pages (in seconds) |
| 5 | `ProductRelated` | Integer | 0 to 705 | Number of product-related catalog pages visited |
| 6 | `ProductRelated_Duration` | Float | 0.0 to 63,973.5 | Total time spent browsing product-related pages (in seconds) |
| 7 | `BounceRates` | Float | 0.0 to 0.2 | Percentage of visitors who enter the page and bounce without triggering another request |
| 8 | `ExitRates` | Float | 0.0 to 0.2 | Percentage of pageviews to the page that were the last in the session |
| 9 | `PageValues` | Float | 0.0 to 361.76 | Average Google Analytics value of pages visited by the user before completing a transaction |
| 10 | `SpecialDay` | Float | 0.0 to 1.0 | Closeness of browsing site visit to a specific holiday or festival (e.g. Valentine's Day, Mother's Day) |
| 11 | `Month` | Categorical | Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec | Calendar month of the browsing session |
| 12 | `OperatingSystems` | Integer / Discrete | 1 to 8 | Operating system identifier used by the visitor |
| 13 | `Browser` | Integer / Discrete | 1 to 13 | Web browser identifier used by the visitor |
| 14 | `Region` | Integer / Discrete | 1 to 9 | Geographic region code of the visitor |
| 15 | `TrafficType` | Integer / Discrete | 1 to 20 | Web traffic acquisition source type (direct, search, referral, banner, email) |
| 16 | `VisitorType` | Categorical | `Returning_Visitor`, `New_Visitor`, `Other` | Customer visitor categorization |
| 17 | `Weekend` | Boolean | `True`, `False` | Boolean flag indicating whether the session occurred on a weekend |
| 18 | `Revenue` | Boolean Target | `True`, `False` | Target variable: Did the session culminate in a purchase? |

---

## 5. Download and Verification Instructions
To reproduce data fetching programmatically:
```python
import urllib.request
import zipfile
import io
import pandas as pd

url = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    df = pd.read_csv(zf.open("online_shoppers_intention.csv"))
df.to_csv("data/online_shoppers_intention.csv", index=False)
```
Delimited by standard comma (`,`).
