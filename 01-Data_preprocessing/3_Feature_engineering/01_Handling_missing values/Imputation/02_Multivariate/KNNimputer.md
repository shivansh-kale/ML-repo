## 🧮 KNN Imputer — Quick Notes

### 1️⃣ When to Use KNN Imputer
- Missing values are **not completely random**
- Features are **correlated**
- Dataset size is **small to medium**

📌 Works better when similar rows exist.

---

### 2️⃣ When NOT to Use It
- Very large datasets (slow)
- High-dimensional data
- Features on very different scales (without scaling)

❌ Distance becomes meaningless.

---

### 3️⃣ Scaling Is Mandatory
- Always scale features before KNN imputation
- Distance-based method → scale affects neighbors

📌 Unscaled data = wrong neighbors.

---

### 4️⃣ Choice of `k` Matters
- Small `k` → sensitive to noise
- Large `k` → over-smoothing

📌 Start with `k = 5` and tune.

---

### 5️⃣ Risk of Data Leakage
- Fit imputer **only on training data**
- Apply same transformation to test data

❌ Never fit on full dataset.

---

### 6️⃣ Validate the Result
- Compare distributions before and after
- Check variance shrinkage
- Watch for unrealistic values

---

### 🧠 Key Reminder
**KNN Imputer trades simplicity for structure—use it only when similarity makes sense.**