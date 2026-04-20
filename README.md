
---

## Pipeline

### 1. Data Preprocessing
- Dropped attention-check columns → **317 columns**
- Removed columns with >50% missing values → **192 columns**
- Median imputation for numeric, mode imputation for categorical
- One-hot encoding → **540 columns**
- Variance threshold filter → **538 final features**

### 2. Exploratory Data Analysis (EDA)
- Country and age distribution of respondents
- Trust towards AI (aipi5) response distribution
- AI Risk Score distribution (composite of aipi1–aipi6)

### 3. AI Risk Score Creation
Composite score calculated from the 6 AI perception items (`aipi1`–`aipi6`) using mean normalisation.

### 4. K-Means Clustering
- Features used: `aipi1`–`aipi6`
- Optimal K selected via Elbow Method → **K = 4**
- PCA used for 2D visualisation

| Cluster | Label | n |
|---------|-------|---|
| 0 | Low Risk Awareness | 8,616 |
| 1 | Moderate Risk Awareness | 4,313 |
| 2 | High Risk Awareness | 5,052 |
| 3 | Mixed Risk Awareness | 5,800 |

### 5. Classification Models

| Model | Accuracy | CV Accuracy (5-fold) |
|-------|----------|----------------------|
| Random Forest | **85.8%** | **85.7%** |
| Logistic Regression | 49.9% | 49.9% |

- Random Forest: 200 estimators, max_depth=15, stratified 80/20 split
- Logistic Regression: Multinomial, max_iter=2000, StandardScaler

### 6. SHAP Feature Importance
- TreeExplainer applied to Random Forest (`check_additivity=False`)
- Top predictors: `respondent_yob`, `trust_AI_1`, `trust_AI_3`, `Ideology_1`, `trust_AI_2`

### 7. Cross-Validation
- Stratified K-Fold (k=5) applied to both models
- RF consistently outperforms LR across all folds

---

## Key Findings

1. **Random Forest significantly outperforms Logistic Regression** (85.8% vs 49.9%), confirming non-linear relationships in the data
2. **4 distinct perception groups** identified — Low Risk is the largest group (36%)
3. **Trust in AI and age** are the strongest predictors of risk awareness group
4. **RF cross-validation is stable** across folds (85.4%–86.1%) — no overfitting
5. **Moderate concern** is the dominant public sentiment (AI Risk Score peaks at 1.0–2.0)

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap plotly

