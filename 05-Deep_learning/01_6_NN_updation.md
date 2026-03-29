# Neural Networks — Gradient Dimensions and Weight Update Intuition

This guide explains one of the most confusing parts of neural networks:

- Why gradient matrices have specific dimensions
- How matrix multiplication is used in backpropagation
- How weight updates work
- Where weights are stored in the network

The goal is to build **clear intuition for the math happening during training**.

---

# 1. Example Neural Network

Suppose we have a small neural network:

```
Input features = 2
Hidden neurons = 3
Output neurons = 1
Batch size = 2
```

Architecture:

```
X → Hidden Layer → Output Layer
```

---

# 2. Input Matrix

Example batch input:

```
X =
[[x11 x12]
 [x21 x22]]
```

Shape:

```
(2 × 2)
```

Meaning:

| Rows | Columns |
|-----|------|
| examples | features |

So:

```
2 examples
2 features
```

---

# 3. First Layer Weights

The hidden layer has **3 neurons**, each connected to **2 input features**.

Weight matrix:

```
W1 =
[[w11 w12 w13]
 [w21 w22 w23]]
```

Shape:

```
(2 × 3)
```

Interpretation:

| Rows | Columns |
|------|------|
| input features | neurons |

Each column represents **one neuron**.

Example:

Neuron 1 weights:

```
[w11, w21]
```

Neuron 2 weights:

```
[w12, w22]
```

Neuron 3 weights:

```
[w13, w23]
```

---

# 4. Forward Pass

Forward propagation uses matrix multiplication.

```
Z1 = X × W1
```

Matrix dimensions:

```
(2×2) × (2×3) = (2×3)
```

Result:

```
Z1 =
[[z11 z12 z13]
 [z21 z22 z23]]
```

Meaning:

```
2 examples
3 neuron outputs
```

Each row represents the output of the hidden layer for one example.

---

# 5. Activation

Apply activation function:

```
A1 = sigmoid(Z1)
```

or

```
A1 = ReLU(Z1)
```

Shape remains:

```
(2 × 3)
```

---

# 6. Output Layer

Output layer has **1 neuron**.

Weights:

```
W2 shape = (3 × 1)
```

Forward pass:

```
Z2 = A1 × W2
```

Dimensions:

```
(2×3) × (3×1) = (2×1)
```

Output:

```
2 predictions
```

Example:

```
[[0.71],
 [0.53]]
```

---

# 7. Loss Calculation

Loss measures prediction error.

Example labels:

```
y =
[[1],
 [0]]
```

Binary cross entropy:

```
loss = -[y log(p) + (1-y) log(1-p)]
```

Loss is calculated **for each example**.

Example:

```
loss1
loss2
```

Then averaged:

```
batch_loss = mean(loss1, loss2)
```

So each batch produces **one loss value**.

---

# 8. Backpropagation Begins

Backpropagation computes gradients.

Gradient meaning:

```
how much the loss changes if a weight changes
```

Mathematically:

```
∂Loss / ∂Weight
```

This tells us:

```
how much each weight should change
```

---

# 9. Gradient of Output Layer Weights

Recall:

```
W2 shape = (3 × 1)
```

Therefore gradient must be:

```
dW2 shape = (3 × 1)
```

Why?

Because weight update rule is:

```
W2_new = W2 − learning_rate * dW2
```

Both matrices must have **identical dimensions**.

Otherwise subtraction would be impossible.

---

# 10. How dW2 is Computed

Gradient formula:

```
dW2 = A1ᵀ × dZ2
```

Matrix shapes:

```
A1 shape = (2 × 3)
A1ᵀ shape = (3 × 2)

dZ2 shape = (2 × 1)
```

Multiplication:

```
(3×2) × (2×1) = (3×1)
```

Perfect — this matches the weight matrix `W2`.

This is why **matrix multiplication is used in backpropagation**.

It aggregates gradient contributions from **all examples in the batch**.

---

# 11. Gradient of Hidden Layer Weights

Hidden layer weights:

```
W1 shape = (2 × 3)
```

Gradient must match:

```
dW1 shape = (2 × 3)
```

Formula:

```
dW1 = Xᵀ × dZ1
```

Matrix shapes:

```
X shape = (2 × 2)
Xᵀ shape = (2 × 2)

dZ1 shape = (2 × 3)
```

Multiplication:

```
(2×2) × (2×3) = (2×3)
```

Again matches `W1`.

---

# 12. Why Matrix Multiplication is Used

Matrix multiplication does two critical things.

### Combine Gradients From All Examples

Each example produces a gradient.

Matrix multiplication efficiently **sums gradients across the batch**.

---

### Maintain Correct Dimensions

Multiplication ensures:

```
gradient shape = weight shape
```

Which allows valid weight updates.

---

# 13. Weight Update

Weights are updated using gradient descent.

Update rule:

```
W = W − learning_rate × gradient
```

Example:

```
W1_new = W1 − η * dW1
W2_new = W2 − η * dW2
```

Where:

```
η = learning rate
```

This step moves the model **slightly toward lower error**.

---

# 14. Where Are Weights Stored?

Yes — neurons effectively **store weights internally**.

In frameworks like Keras:

Each layer stores:

```
weights
bias
```

Example layer:

```python
Dense(3)
```

Internally stores:

```
W1
b1
```

These parameters are **updated after every batch**.

The neuron itself does not store them individually —  
they are stored collectively in the **layer weight matrix**.

---

# 15. Visualizing Weight Matrix

```
Layer weight matrix

      neuron1 neuron2 neuron3
f1 →   w11     w12     w13
f2 →   w21     w22     w23
```

Rows represent:

```
input features
```

Columns represent:

```
neurons
```

During training:

```
each element adjusts slightly
```

Example update:

```
w11 → w11 − η * gradient
```

---

# 16. Key Insight About Gradients

Gradients answer this question:

```
How much did this weight contribute to the loss?
```

If gradient is:

```
positive → decrease weight
negative → increase weight
```

So training becomes a process of:

```
many small corrections
```

---

# 17. Complete Matrix Flow

Forward pass:

```
X → Z1 = XW1 → A1 → Z2 = A1W2 → prediction
```

Backward pass:

```
loss
 ↓
dZ2
 ↓
dW2 = A1ᵀ dZ2
 ↓
dZ1
 ↓
dW1 = Xᵀ dZ1
```

Update:

```
W1 = W1 − η dW1
W2 = W2 − η dW2
```

---

# 18. Big Mental Model

Neural network training consists of:

```
Forward pass → matrix multiplication
Backward pass → matrix multiplication
Weight update → subtraction
```

So everything ultimately reduces to:

```
linear algebra operations on matrices
```

---

# 19. Final Intuition

Each weight is constantly asking:

```
Did increasing me increase the loss or decrease it?
```

The gradient answers that question.

Then the update rule adjusts the weight accordingly.

After **millions of small adjustments**, the network learns useful patterns.