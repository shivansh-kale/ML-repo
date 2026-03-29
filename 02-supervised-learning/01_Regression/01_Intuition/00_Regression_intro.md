# Regression Fundamentals

## 1. What is Regression?
Regression is a **supervised learning technique** used to model the relationship between **input features (X)** and a **continuous target variable (y)**.

**Objective:**  
Learn a function that predicts a numeric value by minimizing prediction error.

**Common Use Cases:**
- House price prediction
- Salary estimation
- Sales forecasting
- Energy consumption prediction

---

## 2. Types of Regression Problems

### 2.1 Simple vs Multiple Regression

| Aspect | Simple Linear Regression | Multiple Linear Regression |
|------|--------------------------|----------------------------|
| Number of features | 1 | More than 1 |
| Equation | \( y = mx + c \) | \( y = w_1x_1 + w_2x_2 + ... + w_nx_n + b \) |
| Complexity | Low | Higher |
| Use case | Basic relationships | Real-world datasets |

---

### 2.2 Linear vs Non-Linear Regression

| Aspect | Linear Regression | Non-Linear Regression |
|------|------------------|----------------------|
| Relationship | Linear | Non-linear |
| Model form | Linear in parameters | Non-linear |
| Interpretability | High | Lower |
| Examples | Linear, Ridge, Lasso | Polynomial, SVR, Tree-based |

📌 **Note:** Polynomial regression is **linear in parameters**, but non-linear in features.

---

## 3. Assumptions of Linear Regression
Linear regression performs best when the following assumptions are reasonably satisfied.

### 3.1 Linearity
- Relationship between predictors and target is linear
- Violation leads to **underfitting**

---

### 3.2 Independence
- Observations are independent
- Commonly violated in **time-series data**

---

### 3.3 Homoscedasticity
- Error variance is constant across all values of X
- Violation (heteroscedasticity) leads to unreliable confidence intervals

---

### 3.4 Normality of Errors
- Residuals follow a normal distribution
- Important for statistical inference
- Less critical for prediction accuracy

---

### 3.5 No Multicollinearity
- Predictors should not be highly correlated
- High multicollinearity causes unstable coefficients

---

### Assumptions Summary Table

| Assumption | Meaning | Violation Effect |
|----------|--------|------------------|
| Linearity | Linear X–y relationship | Underfitting |
| Independence | Observations unrelated | Biased estimates |
| Homoscedasticity | Constant error variance | Invalid confidence intervals |
| Normality | Errors normally distributed | Weak hypothesis tests |
| No Multicollinearity | Low feature correlation | Unstable coefficients |

---

## 4. Cost / Loss Functions
Loss functions quantify the error between **actual** and **predicted** values.

### 4.1 Mean Squared Error (MSE)

\[
MSE = \frac{1}{n} \sum (y - \hat{y})^2
\]

- Penalizes large errors heavily
- Smooth and differentiable

---

### 4.2 Mean Absolute Error (MAE)

\[
MAE = \frac{1}{n} \sum |y - \hat{y}|
\]

- Robust to outliers
- Not differentiable at zero

---

### 4.3 Root Mean Squared Error (RMSE)

\[
RMSE = \sqrt{MSE}
\]

- Same unit as target variable
- Easier to interpret than MSE

---

### Loss Function Comparison

| Metric | Outlier Sensitivity | Differentiable | Interpretation |
|------|--------------------|---------------|----------------|
| MAE | Low | ❌ No | Average absolute error |
| MSE | High | ✅ Yes | Squared error penalty |
| RMSE | High | ✅ Yes | Error in target units |

---

## 5. Bias–Variance Tradeoff (Regression View)

| Aspect | High Bias | High Variance |
|------|----------|---------------|
| Model complexity | Too simple | Too complex |
| Error type | Underfitting | Overfitting |
| Training error | High | Low |
| Test error | High | High |
| Example | Linear model on non-linear data | High-degree polynomial |

📌 **Goal:** Achieve optimal balance between bias and variance.

📌 **Regularization** reduces variance at the cost of slightly increased bias.

---

## 6. Key Takeaways
- Regression predicts **continuous values**
- Linear regression relies on strict assumptions
- Choice of loss function impacts training behavior
- Bias–variance tradeoff explains model performance
- Strong fundamentals are essential before advanced regression models