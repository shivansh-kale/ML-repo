## 🔧 Feature Engineering

Feature Engineering is the process of transforming raw data into meaningful features that help machine learning models learn better patterns.

> Better features often matter more than better models.

---

## Subparts of Feature Engineering

### 1️⃣ Feature Creation
Creating new features from existing data.

- Extracting components from dates (day, month, weekday)
- Creating ratios or aggregates
- Text-based features (word counts, TF-IDF)
- Domain-specific logic

**Goal:** Reveal hidden information in the data.

---

### 2️⃣ Feature Transformation
Modifying feature values while preserving their meaning.

- Scaling (Standardization, Min-Max Scaling)
- Normalization
- Log or power transformations for skewed data

**Why it matters:** Many models assume features are on similar scales.

---

### 3️⃣ Feature Encoding
Converting categorical variables into numerical form.

- Label Encoding
- One-Hot Encoding
- Target / Frequency Encoding

**Note:** Encoding choice can significantly impact model performance.

---

### 4️⃣ Feature Selection
Selecting the most relevant features for modeling.

- Correlation-based selection
- Statistical tests
- Model-based feature importance

**Benefit:** Reduces overfitting and improves efficiency.

---

### 5️⃣ Handling Missing Values
Ensuring data consistency and usability.

- Mean / Median / Mode imputation
- Forward or backward fill
- Model-based imputation

**Insight:** Missing values often carry useful information.

---

### 6️⃣ Handling Outliers
Managing extreme or unusual values.

- Capping or clipping
- Transformation
- Removal (used cautiously)

**Decision:** Always guided by domain knowledge.

---

### 7️⃣ Feature Interaction
Combining features to capture relationships.

- Polynomial features
- Feature crosses
- Ratios and combinations

**Use case:** Especially effective with linear models.

---

## 🧠 Key Principles

- Feature engineering is model-dependent
- Domain knowledge beats complex techniques
- Always validate using cross-validation
- Performed after EDA and before modeling

---

**One-liner:**  
Feature Engineering bridges data understanding and model performance.