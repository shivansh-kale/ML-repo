# Neural Networks — Mathematical and Architectural Concepts

This document explains the **core mathematical and architectural concepts behind neural networks** so that you can confidently understand:

- What happens inside a neural network
- Why each component exists
- What you can modify when designing models

The focus is on **intuition + mathematical structure**, not heavy derivations.

---

# 1. Neural Networks as Function Approximators

A neural network is essentially a **parameterized function**.

```
f(x; θ)
```

Where:

- `x` = input
- `θ` = parameters (weights and biases)

The goal of training is to find parameters `θ` that minimize prediction error.

---

# 2. Linear Model

A simple linear model looks like:

```
y = wx + b
```

or in vector form:

```
y = w · x + b
```

Where:

- `w` = weight vector
- `x` = input vector
- `b` = bias

This is **linear regression**.

Limitation:

A linear model can only represent **linear relationships**.

Example:

```
y = 2x + 1
```

---

# 3. Logistic Regression

For classification we use a **sigmoid transformation**.

```
z = w · x + b
```

Prediction:

```
ŷ = sigmoid(z)
```

Sigmoid function:

```
σ(z) = 1 / (1 + e^(-z))
```

This converts values to **probabilities between 0 and 1**.

Interpretation:

```
ŷ > 0.5 → class 1
ŷ ≤ 0.5 → class 0
```

---

# 4. Neural Networks as Stacked Logistic Regressions

A neural network is simply **multiple logistic/linear models stacked together**.

Example architecture:

```
Input → Linear → Activation → Linear → Activation → Output
```

Mathematically:

```
z¹ = W¹x + b¹
a¹ = activation(z¹)

z² = W²a¹ + b²
a² = activation(z²)
```

Each layer transforms the representation of the data.

---

# 5. Why Do We Need Activation Functions?

Without activation functions, a neural network becomes **just a linear model**.

Example:

```
z1 = W1x
z2 = W2z1
```

Combine:

```
z2 = (W2W1)x
```

This is still **linear**.

Therefore multiple layers would collapse into **one linear transformation**.

Activation functions introduce **non-linearity**, allowing networks to learn complex patterns.

---

# 6. Common Activation Functions

## Sigmoid

```
σ(z) = 1 / (1 + e^-z)
```

Range:

```
(0,1)
```

Used for:

- Binary classification output

Problem:

```
vanishing gradients
```

---

## Tanh

```
tanh(z)
```

Range:

```
(-1,1)
```

Better centered around zero.

Still suffers from gradient vanishing.

---

## ReLU

```
ReLU(z) = max(0, z)
```

Advantages:

- Simple
- Efficient
- Reduces vanishing gradient problem

Problem:

```
dead neurons
```

---

## Leaky ReLU

```
LeakyReLU(z) = max(0.01z, z)
```

Fixes dead neuron problem.

---

# 7. The Output Layer

The output layer structure depends on the **task type**.

---

## Regression Output

No activation or linear activation.

```
y = Wz + b
```

Example Keras:

```python
Dense(1)
```

---

## Binary Classification Output

Use sigmoid.

```python
Dense(1, activation="sigmoid")
```

Output:

```
P(class = 1)
```

---

## Multi-Class Classification Output

Use **Softmax**.

---

# 8. Softmax Function

Softmax converts logits into probabilities.

Logits:

```
z = [z1, z2, z3]
```

Softmax:

```
softmax(zi) = e^zi / Σ e^zj
```

Properties:

- All probabilities sum to 1
- Output represents probability distribution

Example:

```
logits = [2.0, 1.0, 0.1]

softmax → [0.65, 0.24, 0.11]
```

---

# 9. Logits

Logits are **raw outputs of the neural network before activation**.

Example:

```
z = W x + b
```

These values are **not probabilities**.

Frameworks often allow using logits directly for numerical stability.

Example:

```python
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
```

Advantages:

- avoids numerical overflow
- improves stability

---

# 10. Vanishing Gradient Problem

During backpropagation gradients propagate through layers.

In deep networks:

```
gradient = derivative1 × derivative2 × derivative3 ...
```

If derivatives are small (<1):

```
gradients shrink exponentially
```

Result:

```
earlier layers stop learning
```

This is called **vanishing gradient**.

Common with:

- sigmoid
- tanh

Solutions:

- ReLU activation
- Batch normalization
- Residual connections

---

# 11. Exploding Gradient Problem

Opposite of vanishing gradients.

If derivatives >1:

```
gradients grow exponentially
```

Result:

- unstable training
- weights become huge

Solutions:

- gradient clipping
- better initialization
- normalization layers

---

# 12. Weight Initialization

Initialization is important for stable training.

---

## Xavier Initialization

Used for:

```
tanh
sigmoid
```

Variance:

```
1 / n_inputs
```

---

## He Initialization

Used for:

```
ReLU
```

Variance:

```
2 / n_inputs
```

Example Keras:

```python
Dense(128, kernel_initializer="he_normal")
```

---

# 13. Batch Normalization

Batch normalization normalizes activations.

```
x̂ = (x - μ) / σ
```

Then scale and shift:

```
y = γx̂ + β
```

Benefits:

- faster training
- stable gradients
- higher learning rates possible

Example:

```python
BatchNormalization()
```

---

# 14. Bias Terms

Bias allows the network to shift the activation function.

Without bias:

```
y = Wx
```

With bias:

```
y = Wx + b
```

Bias increases model flexibility.

---

# 15. Depth vs Width

### Width

Number of neurons per layer.

### Depth

Number of layers.

Deep networks learn **hierarchical representations**.

Example in vision:

```
edges → textures → shapes → objects
```

---

# 16. Universal Approximation Theorem

This theorem states:

A neural network with **one hidden layer** can approximate **any continuous function**, given enough neurons.

However:

Deep networks are more **efficient** than shallow ones.

---

# 17. Keras Example Model

Example neural network:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation="relu", input_shape=(10,)),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

Explanation:

```
Input: 10 features
Hidden layer: 64 neurons
Hidden layer: 64 neurons
Output: probability
```

---

# 18. Model with Softmax Output

Example multi-class classifier.

```python
model = Sequential([
    Dense(128, activation="relu", input_shape=(20,)),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])
```

Used for problems like:

```
digit classification
image classification
```

---

# 19. Logits-Based Training Example

Instead of softmax activation:

```python
Dense(10)
```

Loss function:

```python
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
```

The framework internally applies softmax.

---

# 20. Key Concepts to Master

To deeply understand neural networks, one should be comfortable with:

- forward propagation
- backpropagation
- activation functions
- gradient flow
- initialization
- optimization
- architecture design
- output layer design
- loss functions

These concepts allow practitioners to **modify architectures intelligently and debug training problems**.

---

# Final Insight

A neural network is fundamentally:

```
Linear transformations + Non-linear activations
```

Stacking many such transformations allows the network to learn **complex functions from data**.

Understanding the **mathematical and architectural design choices** is essential for building robust deep learning systems.