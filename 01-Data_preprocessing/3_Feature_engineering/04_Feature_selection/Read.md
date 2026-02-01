# Feature Selection

## What is Feature Selection?
Process of selecting the **most relevant features** and removing:
- Irrelevant features
- Redundant features
to improve model performance.

---

## Why Feature Selection?
- Reduces overfitting
- Improves model accuracy
- Faster training & inference
- Better interpretability

---

## Types of Feature Selection

### 1. Filter Methods
Select features based on **statistical measures**
- Correlation
- Chi-Square test
- ANOVA
- Mutual Information

📌 Model-independent, fast

---

### 2. Wrapper Methods
Use model performance to select features
- Forward Selection
- Backward Elimination
- Recursive Feature Elimination (RFE)

📌 More accurate, computationally expensive

---

### 3. Embedded Methods
Feature selection happens **during model training**
- Lasso (L1 Regularization)
- Decision Trees
- Random Forest Feature Importance

📌 Good balance of speed & accuracy

---

## Common Techniques

### Correlation Threshold
- Remove highly correlated features
- Avoid multicollinearity

---

### L1 Regularization (Lasso)
- Shrinks some coefficients to zero
- Performs automatic feature selection

---

### Tree-Based Importance
- Based on information gain / impurity reduction

---

## Feature Selection vs Feature Extraction
- **Selection** → keeps original features
- **Extraction** → creates new features (e.g., PCA)

---

## Best Practices
- Perform selection after train-test split
- Use cross-validation
- Combine domain knowledge + statistics
- Avoid data leakage

---

## Quick Summary
- Filter → fast, simple
- Wrapper → accurate, slow
- Embedded → efficient, practical
