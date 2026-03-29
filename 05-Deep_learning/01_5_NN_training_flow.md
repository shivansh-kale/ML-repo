# Neural Network Training Lifecycle (Step-by-Step)

This document explains **how a neural network processes data during training**, including:

- How input flows through the network
- How predictions are made
- How loss is calculated
- How gradients move backward
- How weights are updated
- How batches and epochs work

The goal is to **build a strong mental visualization of the entire neural network training process**.

---

# 1. High-Level Overview of Neural Network Training

Training a neural network follows a repeating cycle:

```
Data → Forward Pass → Loss Calculation → Backpropagation → Weight Update
```

This loop repeats **many times** until the model learns patterns in the data.

The process happens across multiple **batches** and **epochs**.

---

# 2. Example Dataset

Suppose our dataset looks like this:

```
X = [
 [200,17],
 [120,5],
 [425,20],
 [212,18]
]
```

Shape:

```
4 × 2
```

Meaning:

| Row | Example | Features |
|----|----|----|
|1|sample 1|2 features|
|2|sample 2|2 features|
|3|sample 3|2 features|
|4|sample 4|2 features|

Labels:

```
y = [1,0,0,1]
```

Where:

```
1 = class 1
0 = class 0
```

---

# 3. What is a Batch?

A **batch** is a small subset of the dataset used during one training step.

Example:

```
dataset size = 1000
batch size = 32
```

The network processes:

```
32 samples at once
```

Benefits:

- faster training
- stable gradient updates
- efficient GPU usage

Example batch:

```
Batch 1

[[200,17],
 [120,5]]
```

```
Batch 2

[[425,20],
 [212,18]]
```

Each batch goes through **forward and backward propagation**.

---

# 4. What is an Epoch?

An **epoch** means the model has seen the **entire dataset once**.

Example:

```
dataset size = 1000
batch size = 100
```

Then:

```
1000 / 100 = 10 batches
```

Therefore:

```
1 epoch = 10 training steps
```

Training often runs for many epochs:

```
epochs = 50
```

Meaning the model sees the **dataset 50 times**.

---

# 5. Forward Propagation (Input → Prediction)

Forward propagation means:

> passing the input through the neural network to generate predictions.

Example network:

```
Input Layer → Hidden Layer → Output Layer
```

---

## Step 1: Input Enters Network

Example batch:

```
X_batch =
[[200,17],
 [120,5]]
```

Shape:

```
2 × 2
```

Meaning:

```
2 examples
2 features each
```

---

## Step 2: First Layer Computation

Suppose the hidden layer has **3 neurons**.

Weights:

```
W1 shape = (2 × 3)
```

Example weight matrix:

```
W1 =
[[w11 w12 w13]
 [w21 w22 w23]]
```

Matrix multiplication:

```
Z1 = X_batch × W1
```

Shape calculation:

```
(2×2) × (2×3) = (2×3)
```

Output:

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

---

## Step 3: Activation Function

Activation introduces **non-linearity**.

Example:

```
A1 = sigmoid(Z1)
```

or

```
A1 = ReLU(Z1)
```

Shape remains:

```
2 × 3
```

Now the data representation has been transformed.

---

## Step 4: Output Layer

Suppose the final layer has **1 neuron**.

Weights:

```
W2 shape = (3 × 1)
```

Matrix multiplication:

```
Z2 = A1 × W2
```

Shape:

```
(2×3) × (3×1) = (2×1)
```

Example:

```
Z2 =
[[0.91],
 [0.14]]
```

Apply sigmoid:

```
Predictions =
[[0.71],
 [0.53]]
```

These values represent **predicted probabilities**.

---

# 6. Loss Calculation

Loss measures **how wrong the predictions are**.

Example true labels:

```
y =
[[1],
 [0]]
```

Binary Cross-Entropy:

```
loss = -[y log(p) + (1-y) log(1-p)]
```

Loss is computed **for each example**.

Example:

```
loss1 = error of sample 1
loss2 = error of sample 2
```

Then we compute the **average loss of the batch**:

```
batch_loss = mean(loss1, loss2)
```

So each batch produces:

```
1 loss value
```

---

# 7. Backpropagation (Moving Backward)

Backpropagation calculates **how each weight contributed to the loss**.

It computes gradients:

```
∂Loss / ∂Weight
```

This tells us:

```
how much each weight should change
```

Gradients are calculated using the **chain rule**.

---

## Gradient Shapes

Example:

```
W2 shape = (3 × 1)
```

Gradient:

```
dW2 shape = (3 × 1)
```

Hidden layer gradient:

```
W1 shape = (2 × 3)
```

Gradient:

```
dW1 shape = (2 × 3)
```

These gradients are computed using **matrix operations**.

---

# 8. Weight Update

Weights are updated using **gradient descent**.

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

This step makes the network **slightly better at predicting**.

---

# 9. Next Batch Enters

After updating weights, the next batch is processed.

Example:

```
[[425,20],
 [212,18]]
```

The same steps repeat:

```
Forward pass
Loss calculation
Backpropagation
Weight update
```

---

# 10. End of an Epoch

After all batches are processed:

```
1 epoch completed
```

Example:

```
dataset = 1000
batch size = 100
```

```
1 epoch = 10 weight updates
```

---

# 11. Next Epoch Begins

The network **keeps the updated weights** and processes the dataset again.

Example training progress:

```
Epoch 1 → loss = 0.65
Epoch 2 → loss = 0.42
Epoch 3 → loss = 0.31
```

The model gradually improves.

---

# 12. Full Neural Network Training Flow

```
Dataset
   ↓
Split into batches
   ↓
Batch enters network
   ↓
Forward propagation
   ↓
Predictions
   ↓
Loss calculation
   ↓
Backpropagation
   ↓
Gradient calculation
   ↓
Weight update
   ↓
Next batch
```

Repeat until:

```
All batches processed → epoch finished
```

Then start the **next epoch**.

---

# 13. Important Insight

Weights are updated:

```
after each batch
```

NOT after each epoch.

An epoch is simply **one complete pass over the dataset**.

---

# 14. Intuitive Visualization

Think of training like **studying for an exam**.

Dataset:

```
all questions in the textbook
```

Batch:

```
a small group of questions
```

Epoch:

```
one full revision of the entire textbook
```

Each time you revise:

```
you adjust your understanding slightly
```

The neural network adjusts **weights instead of knowledge**.

---

# 15. Core Mathematical Insight

Neural networks are mostly:

```
matrix multiplications
+ activation functions
+ gradient calculations
```

Forward pass:

```
matrix multiplication
```

Backward pass:

```
matrix multiplication with gradients
```

Weight update:

```
subtraction operation
```

Everything ultimately reduces to **linear algebra operations on matrices**.

---

# Final Key Idea

A neural network learns by **repeatedly adjusting weights to reduce prediction error**.

Training is simply:

```
Forward pass → Loss → Backpropagation → Update weights
```

repeated across:

```
batches
epochs
```

until the model learns useful patterns from the data.