# Feature Scaling — Compact & Clear Notes

## One-Line Memory Hooks

- **Scaling** → "Make features comparable"
- **Normalization** → "Force into range"
- **Standardization** → "Center & spread"
- **Transformation** → "Fix shape"

---


## 1. What is Feature Scaling?
Feature scaling is the process of **bringing numeric features to a comparable range** so that:
- No feature dominates others due to large magnitude
- Optimization algorithms converge faster
- Distance- and gradient-based models behave correctly

---

## 2. Scaling vs Normalization vs Mathematical Transformation

| Term | What it does | Purpose |
|----|----|----|
| **Scaling** | Changes the **range or spread** of data | Make features comparable |
| **Normalization** | Special type of scaling to a **fixed range** | Bound values (usually 0–1) |
| **Mathematical Transformation** | Changes the **shape/distribution** | Reduce skew, stabilize variance |

> Scaling ≠ Normalization  
> Normalization ⊂ Scaling  
> Transformations are **not scaling**, but often used **before scaling**

---

## 3. Feature Scaling (General)

### Definition
Rescales values **without changing distribution shape**.

### Why needed?
- Distance-based models are scale-sensitive
- Gradient descent converges faster
- Regularization behaves correctly

### Algorithms that NEED scaling
- KNN
- K-Means
- SVM
- Logistic Regression
- Linear Regression (GD-based)
- Neural Networks

### Algorithms that DON'T need scaling
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

(They split on feature thresholds, not distances)

---

--- 

## 4. Normalization (Min–Max Scaling)

### Formula    
- `x' = (x − min) / (max − min)`

### Output Range
- `[0,1]` (or custom range)


### When to use
- You need **bounded values**
- Data has **no strong outliers**
- Used in **image processing, neural networks**

### Pros
- Preserves relative distances
- Intuitive scale

### Cons
- **Highly sensitive to outliers**

---

---


## 5. Standardization (Z-score Scaling)

### Formula
- `x' = (x − mean) / std`


### Output
- `- Mean = 0`
- `- Std = 1`
- `- Range = unbounded`

### When to use
- Data is **roughly normal**
- Algorithms assume Gaussian-like inputs
- Most common default choice

### Pros
- Handles moderate outliers better
- Works well with regularization

### Cons
- Not bounded

---

---


## 6. Robust Scaling

### Formula
- `x' = (x − median) / IQR`

### When to use
- Data contains **strong outliers**
- Median-based robustness needed

### Pros
- Outlier-resistant

### Cons
- Less interpretable scale

---

## 7. Mathematical Transformations (Shape-Changing)

> Transformations change **distribution**, not just scale.

### Why needed?
- Reduce skewness
- Stabilize variance
- Make data closer to normal

---


### Log Transformation

### formula
- `x' = log(x)`  or `log(x+c)`

**Use when**:
- Right-skewed data
- Exponential growth
- Positive values only

---

### Square Root
`x' = √x`
**Use when**:
- Count data
- Mild skew

---


### Box-Cox
- Power transform
- Requires **positive values**
- Automatically finds best power

---

### Yeo-Johnson
- Box-Cox variant
- Works with **zero and negative values**

---


---


## 8. Correct Order (Very Important)
Raw Data
↓
Mathematical Transformation (if skewed)
↓
Scaling / Normalization
↓
Model





❌ Never scale before fixing skew

---

## 9. What to Use — Decision Table

| Data Situation | Best Choice |
|----|----|
| No outliers, bounded needed | Min–Max |
| Roughly normal | Standardization |
| Strong outliers | Robust Scaler |
| Heavy right skew | Log → Standardize |
| Tree-based models | No scaling needed |

---

## 10. Common Mistakes

❌ Scaling target variable without reason  
❌ Fitting scaler on test data  
❌ Scaling categorical features  
❌ Using Min–Max with outliers  
❌ Thinking normalization = distribution fix

---

## 11. One-Line Memory Hooks

- **Scaling** → "Make features comparable"
- **Normalization** → "Force into range"
- **Standardization** → "Center & spread"
- **Transformation** → "Fix shape"

---

## 12.One-Liner

> “Scaling adjusts magnitude, normalization bounds values, and transformations change distribution. Distance-based models require scaling; tree-based models do not.”

---