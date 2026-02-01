## 🧹 Handling Missing Values — Final Notes

- Check **why** data is missing before fixing it
- **Few + random missing values** → drop rows (CCA)
- **Numerical features** → Median > Mean (safe default)
- **Categorical features** → Mode or explicit “Missing”
- **KNN Imputer** → use when similar rows exist (scale first)
- **Always fit on train data only** (avoid leakage)
- Compare **before vs after distributions**
- Missingness itself can be a **feature**

**Rule:** Start simple → validate → increase complexity only if needed.