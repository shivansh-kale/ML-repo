# Regression — Big Picture Overview & Decision Framework

> This file defines *how to think about regression as a space of models*, not how to implement them.
> The goal is to understand **why different regression algorithms exist** and **when each one is the right tool**.

---

## 1. What Regression Is Really About

At its core, regression answers one question:

> **How does a numeric outcome change as inputs change?**

Every regression algorithm differs only in:
1. **What kind of relationships it can express**
2. **How it balances bias vs variance**
3. **What assumptions it makes about data**
4. **How much interpretability it gives up for performance**

---

## 2. High-Level Classification of Regression Models

Regression algorithms can be grouped by **how they model relationships**.

| Category | Core Idea | Examples |
|-------|----------|----------|
| Linear models | Assume linear relationship in parameters | Linear, Ridge, Lasso |
| Basis expansion models | Transform features to capture non-linearity | Polynomial |
| Distance-based models | Predict using nearby points | KNN Regression |
| Margin-based models | Fit within tolerance margins | SVR |
| Tree-based models | Split feature space into regions | Decision Tree |
| Ensemble models | Combine many weak models | Random Forest, Gradient Boosting |
| Probabilistic models | Predict distributions, not just values | Bayesian Regression |

---

## 3. Linear Family of Regression Models

### 3.1 Linear Regression
**Mental model:**  
Project target onto feature space using straight lines / hyperplanes.

**Strengths**
- Highly interpretable
- Fast and stable
- Strong baseline

**Weaknesses**
- Cannot capture complex non-linear patterns
- Sensitive to outliers and multicollinearity

**Use when**
- Relationship is approximately linear
- Interpretability matters
- Dataset is small to medium

---

### 3.2 Ridge Regression (L2)
**Why it exists**
- Linear regression becomes unstable when features are correlated

**What it changes**
- Penalizes large coefficients → smoother solutions

**Use when**
- Multicollinearity is present
- You want stability over sparsity
- All features matter a little

---

### 3.3 Lasso Regression (L1)
**Why it exists**
- Too many features dilute interpretability

**What it changes**
- Forces some coefficients to exactly zero

**Use when**
- Feature selection is important
- Dataset has many irrelevant features

---

### 3.4 Elastic Net
**Why it exists**
- Ridge and Lasso solve different problems

**Use when**
- Many correlated features
- You want both stability and sparsity

---

## 4. Non-Linear Extensions of Linear Models

### 4.1 Polynomial Regression
**Key insight**
- Still linear in parameters, but non-linear in features

**Strength**
- Captures smooth curves

**Risk**
- High-degree polynomials overfit badly

**Use when**
- Relationship is smooth but non-linear
- Dataset size is limited
- You want to stay close to linear models

---

## 5. Instance-Based Regression

### 5.1 KNN Regression
**Mental model**
- “Tell me the average outcome of similar past cases”

**Strengths**
- No training phase
- Captures local patterns

**Weaknesses**
- Poor scalability
- Sensitive to noise and feature scaling

**Use when**
- Dataset is small
- Patterns are local
- Interpretability is not critical

---

## 6. Margin-Based Regression

### 6.1 Support Vector Regression (SVR)
**Mental model**
- Fit a function that stays within an acceptable error margin

**Strengths**
- Effective in high-dimensional spaces
- Robust to outliers (with proper kernel)

**Weaknesses**
- Hard to tune
- Computationally expensive

**Use when**
- Dataset is small to medium
- High-dimensional features
- Clear margin of acceptable error

---

## 7. Tree-Based Regression Models

### 7.1 Decision Tree Regressor
**Mental model**
- Split data into regions and predict averages

**Strengths**
- Captures non-linear interactions
- Easy to visualize

**Weaknesses**
- High variance
- Overfits easily

**Use when**
- Interpretability matters
- Relationships are non-linear
- Baseline non-linear model

---

### 7.2 Random Forest Regressor
**Why it exists**
- Single trees are unstable

**What it changes**
- Averages many trees → variance reduction

**Strengths**
- Strong performance out-of-the-box
- Robust to noise

**Weaknesses**
- Less interpretable
- Large models

**Use when**
- You want strong performance without heavy tuning
- Tabular data

---

### 7.3 Gradient Boosting Regressors
(XGBoost, LightGBM, CatBoost)

**Mental model**
- Sequentially fix mistakes of previous models

**Strengths**
- State-of-the-art for tabular data
- Handles complex patterns

**Weaknesses**
- Sensitive to hyperparameters
- Less interpretable

**Use when**
- Prediction accuracy is top priority
- Medium to large datasets
- Structured/tabular data

---

## 8. Probabilistic Regression

### 8.1 Bayesian Regression
**Mental model**
- Treat parameters as random variables

**Strengths**
- Uncertainty estimation
- Incorporates prior knowledge

**Weaknesses**
- Computationally expensive
- Conceptually complex

**Use when**
- Uncertainty matters
- Data is limited
- Scientific/statistical settings

---

## 9. Regression Algorithm Selection — Decision Table

| Situation | Recommended Model |
|--------|------------------|
| Need interpretability | Linear / Ridge |
| Multicollinearity | Ridge / Elastic Net |
| Feature selection | Lasso |
| Smooth non-linearity | Polynomial |
| Small dataset, local patterns | KNN |
| High-dimensional space | SVR |
| Non-linear interactions | Decision Tree |
| Strong baseline performance | Random Forest |
| Best possible accuracy | Gradient Boosting |
| Uncertainty required | Bayesian Regression |

---

## 10. The One Unifying Mental Model

> Every regression algorithm answers the same question using a different **assumption about structure** and a different **bias–variance compromise**.

- Linear models assume simplicity
- Trees assume piecewise constancy
- Kernels assume similarity
- Ensembles assume many weak truths combine into a strong one

There is no “best regression model” — only a **best assumption for the problem**.

---

## 11. Final Perspective for This Repo

- This overview defines the **map**
- Each future folder defines a **region**
- You don’t learn regression by memorizing algorithms  
  You learn it by understanding **why each algorithm exists**

> Once you can explain *why* a model should work before training it, you are thinking like a real ML engineer.
