##  Handling Missing Numerical Values — Quick Notes

### 1️⃣ Check Why Values Are Missing First
- Random (MCAR) → simple imputation OK
- Systematic (MAR / MNAR) → be careful

📌 Imputation can hide important patterns.

---

### 2️⃣ Choose Strategy Based on Distribution
- **Mean** → symmetric / normal data
- **Median** → skewed data or presence of outliers
- **Mode** → rarely suitable for numerical features

📌 Median is the safest default choice.

---

### 3️⃣ Always Fit on Training Data Only
- Fit imputer on the train set
- Transform train and test sets separately

❌ Never fit on the full dataset (data leakage).

---

### 4️⃣ Compare Before vs After
- Mean / median shift
- Distribution shape
- Effect on outliers

📌 Imputation should not distort the distribution heavily.

---

### 5️⃣ Missingness Itself May Be a Feature
Add a missing-indicator column if:
- Many values are missing
- Missingness is informative

---

### 6️⃣ Simple ≠ Bad
- `SimpleImputer` often outperforms complex methods
- Complex imputers require more data

📌 Start simple, then iterate.