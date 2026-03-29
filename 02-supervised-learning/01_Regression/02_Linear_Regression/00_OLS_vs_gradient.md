# OLS vs Gradient Descent — How Regression Learns

## The Core Question
Once we define a regression model and a loss function, we still need to answer:

> **How do we find the best parameters?**

There are two fundamental approaches:
- **Closed-form solution (OLS)**
- **Iterative optimization (Gradient Descent)**

---

## Ordinary Least Squares (OLS)

### What it is
OLS finds the parameters that **exactly minimize squared error** using linear algebra.

\[
(X^\top X)w = X^\top y
\]

This equation is solved **directly**.

---

### Why OLS Works
- Squared error creates a **convex loss surface**
- Convexity guarantees a **single global minimum**
- Linear algebra gives an exact solution

---

### Why sklearn Uses OLS for LinearRegression
- Fast for small–medium datasets
- Deterministic (no learning rate, no iterations)
- Numerically stable (with internal optimizations)
- No tuning required

📌 `LinearRegression()` in sklearn uses an OLS-based solver under the hood.

---

## Gradient Descent

### What it is
Gradient Descent **iteratively updates parameters** by following the slope of the loss surface.

\[
w := w - \alpha \nabla L(w)
\]

Where:
- \(\alpha\) is the learning rate

---

### Why Gradient Descent Exists
OLS becomes impractical when:
- Dataset is very large
- Feature space is high-dimensional
- Matrix inversion is expensive or unstable

Gradient descent trades **exactness** for **scalability**.

---

## Key Differences

| Aspect | OLS | Gradient Descent |
|----|----|----|
| Solution | Exact | Approximate |
| Method | Matrix algebra | Iterative updates |
| Speed | Fast (small data) | Scales to large data |
| Hyperparameters | None | Learning rate, iterations |
| Determinism | Yes | No |
| Used by sklearn LR | ✅ Yes | ❌ No |

---

## When to Use What

| Situation | Preferred Method |
|--------|----------------|
| Small to medium datasets | OLS |
| Few to moderate features | OLS |
| Need fast, stable baseline | OLS |
| Massive datasets | Gradient Descent |
| Streaming / online learning | Gradient Descent |
| Neural networks | Gradient Descent |

---

## Important Perspective
OLS and Gradient Descent **solve the same objective** —  
they differ only in *how* they reach the solution.

> Optimization choice affects **efficiency**, not the model’s intent.

---

## Key Takeaway
- OLS is about **exact solutions**
- Gradient Descent is about **practical scalability**
- sklearn chooses OLS because it is **simple, fast, and reliable** for linear models

Understanding this distinction prevents confusion when moving from linear models to deep learning.