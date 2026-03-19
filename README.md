# Credit-Risk-Prediction-Loan-Default-Modeling
Machine learning project to predict loan defaults using logistic regression. Exploratory data analysis, feature importance, and credit risk insights from a banking dataset.

# Credit Risk Prediction & Loan Default Analysis

**Objective**  
Build machine learning models to predict loan defaults and understand key drivers of credit risk using customer banking data. The goal is to help banks reduce non-performing loans (NPLs), improve lending decisions, and enhance financial stability.

**Business Problem**  
Loan defaults cause major losses for banks (>20% of assets in severe cases), reduce profitability, damage reputation, and restrict credit availability — harming economic growth. Effective credit risk analytics using data and ML can identify high-risk borrowers early and optimize lending.

**Dataset**  
- Source: Bankloans.csv (Kaggle-inspired / public credit dataset)  
- Rows: 1,150 (after removing ~300 duplicates)  
- Columns (9 features):  
  - **age**: Customer age  
  - **ed**: Education level (1–5)  
  - **employ**: Years of employment  
  - **income**: Annual income ($000s)  
  - **debtinc**: Debt-to-income ratio (%)  
  - **creddebt**: Credit debt ($000s)  
  - **othdebt**: Other debt ($000s)  
  - **address**: Years at current address (dropped — not predictive)  
  - **default**: Target (1 = defaulted, 0 = repaid) — binary  

No missing values after cleaning. Target is imbalanced (~26% defaults).

**Methodology**  
1. **EDA** (Python + pandas, matplotlib/seaborn in Google Colab)  
   - Univariate: Distributions of age, income, debtinc, etc.  
   - Multivariate: Correlation matrix + scatter plots  
   - Key insight: debtinc and creddebt show strongest positive correlation with default.

2. **Logistic Regression** (Classification — Predict default)  
   - Features: age, ed, employ, income, debtinc, creddebt, othdebt  
   - Train/test split: 80/20  
   - Results:

     | Metric    | Value |
     |-----------|-------|
     | Accuracy  | 0.87  |
     | Precision | 0.92  |
     | Recall    | 0.58  |
     | F1-score  | 0.71  |

   - Strong precision → reliable when predicting default (few false alarms).  
   - Moderate recall → misses some defaulters (class imbalance effect).  
   - debtinc was the strongest predictor.

3. **Linear Regression** (Predict debtinc — understand financial strain)  
   - Features: age, ed, employ, income  
   - R² = -0.038 (negative!)  
   - MSE = 47.68  
   - Conclusion: Linear model fails — likely non-linear relationships or missing predictors. Not suitable for prediction.

4. **Hypothesis Test** (Independent t-test)  
   - H₀: Mean income (defaulters) = Mean income (non-defaulters)  
   - p-value = 0.97 → Fail to reject H₀  
   - Income alone is **not** a significant differentiator of default risk.

**Key Findings & Insights**  
- **Debt-to-income ratio (debtinc)** and **credit debt (creddebt)** are the strongest predictors of default — higher values sharply increase risk.  
- Focus lending policies on debtinc thresholds (e.g., flag >15%).  
- Income shows weak/no direct link to default after controlling for debt levels.  
- Logistic regression provides a solid baseline model (87% accuracy) for identifying risky borrowers.  
- Linear regression unsuitable for debtinc prediction.

**Business Impact**  
- Proactive identification of high-risk applicants → fewer defaults, lower provisions for loan losses.  
- Better risk-based pricing and customer segmentation.  
- Improved regulatory compliance and overall portfolio stability.

**Limitations**  
- Small dataset (1,150 rows)
- No external data (credit score, payment history, macro factors).  
- Class imbalance not addressed (e.g., SMOTE).  

**Technologies Used**  
Python, pandas, scikit-learn, matplotlib, seaborn, Google Colab


