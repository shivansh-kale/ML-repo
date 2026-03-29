# Convolutional Neural Network (CNN) Architecture — From First Principles

Since you’re currently learning **Deep Learning and CNNs**, this explanation follows a **systematic architecture view** so you understand **how the data flows through a CNN**, why each component exists, and the **different CNN architecture types used in research and industry**.

---

# 1. What a CNN Actually Does

A **Convolutional Neural Network (CNN)** is a neural network designed for **grid-like data** such as:

* Images (2D grid of pixels)
* Audio spectrograms
* Video frames
* Medical scans
* Satellite imagery

Instead of learning **global weights like a fully connected network**, CNNs learn **local patterns**.

Example patterns:

* edges
* corners
* textures
* shapes
* objects

This happens through **convolution filters**.

---

# 2. High-Level CNN Pipeline

A typical CNN architecture looks like this:

```
Input Image
     ↓
Convolution Layer
     ↓
Activation (ReLU)
     ↓
Pooling Layer
     ↓
Convolution Layer
     ↓
Activation
     ↓
Pooling
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Output Layer
```

Or conceptually:

```
Image → Feature Extraction → Feature Compression → Classification
```

---

# 3. Core Components of CNN Architecture

## 3.1 Input Layer

The input is typically an **image tensor**.

Example:

```
RGB image: 224 × 224 × 3
```

Where:

* 224 = height
* 224 = width
* 3 = color channels (R,G,B)

For grayscale:

```
28 × 28 × 1
```

Example: MNIST digits.

---

# 4. Convolution Layer (Feature Extraction)

This is the **most important part of CNN**.

The convolution layer uses **filters (kernels)** to detect patterns.

Example kernel:

```
3 × 3 filter
```

Example operation:

```
Input Image Patch
[ 1 2 0
  3 1 2
  0 1 3 ]

Kernel
[ 1 0 -1
  1 0 -1
  1 0 -1 ]
```

The kernel **slides across the image** performing:

```
Element-wise multiplication + sum
```

Result = **Feature Map**

---

### Why multiple filters?

Each filter learns a **different pattern**:

* Filter 1 → vertical edges
* Filter 2 → horizontal edges
* Filter 3 → corners
* Filter 4 → textures

So output becomes:

```
224×224×3 image
↓
Conv layer with 32 filters
↓
224×224×32 feature maps
```

---

# 5. Activation Function

After convolution we apply **non-linearity**.

Most common:

```
ReLU(x) = max(0, x)
```

Why?

Without activation the network becomes **linear**, which cannot model complex patterns.

---

# 6. Pooling Layer (Dimensionality Reduction)

Pooling **reduces spatial size**.

Example:

```
Input: 4×4 feature map
```

Max pooling (2×2):

```
[1 3 | 2 1] → max = 3
[4 6 | 5 2] → max = 6
------------
[2 1 | 0 2] → max = 2
[1 2 | 3 4] → max = 4
```

Output:

```
2×2 feature map
```

Benefits:

* reduces computation
* prevents overfitting
* increases receptive field

Common types:

| Type                   | Meaning                     |
| ---------------------- | --------------------------- |
| Max Pooling            | selects strongest feature   |
| Average Pooling        | takes mean                  |
| Global Average Pooling | averages entire feature map |

---

# 7. Stacking Convolution Layers

CNNs become powerful when layers are stacked.

| Layer  | What it learns |
| ------ | -------------- |
| Conv 1 | edges          |
| Conv 2 | corners        |
| Conv 3 | textures       |
| Conv 4 | object parts   |
| Conv 5 | objects        |

Example intuition:

```
Cat image

Conv1 → edges
Conv2 → fur texture
Conv3 → ear shapes
Conv4 → cat face
Conv5 → full cat
```

This is called **hierarchical feature learning**.

---

# 8. Flatten Layer

After feature extraction, the tensor is converted to a vector.

Example:

```
Feature Map = 7 × 7 × 512
```

Flatten:

```
25088 vector
```

---

# 9. Fully Connected Layer

Acts like a **traditional neural network classifier**.

Example:

```
25088
   ↓
4096 neurons
   ↓
4096 neurons
   ↓
1000 output classes
```

Used for classification.

---

# 10. Output Layer

Final prediction.

For classification use **Softmax**:

```
P(class_i) = e^zi / Σ e^zj
```

Example output:

```
Dog → 0.02
Cat → 0.91
Horse → 0.04
Car → 0.03
```

Prediction = **Cat**

---

# 11. Important CNN Hyperparameters

### Kernel Size

Common values:

```
3×3
5×5
7×7
```

Modern CNNs mostly use **3×3 kernels**.

---

### Stride

Stride = step size of convolution.

```
stride = 1 → move 1 pixel
stride = 2 → move 2 pixels
```

Higher stride → smaller feature map.

---

### Padding

Padding adds zeros around the image.

| Type  | Result                    |
| ----- | ------------------------- |
| Valid | no padding                |
| Same  | output size same as input |

---

# 12. Types of CNN Architectures

---

## 12.1 LeNet (1998)

First successful CNN.

Architecture:

```
Input 32×32
↓
Conv
↓
Pooling
↓
Conv
↓
Pooling
↓
FC
↓
Output
```

Used for:

* handwritten digit recognition

Key idea:

```
Conv → Pool → Conv → Pool
```

---

## 12.2 AlexNet (2012)

Architecture that **started the deep learning revolution**.

Key features:

* ReLU activation
* dropout
* GPU training
* deeper network

Architecture:

```
Conv
Conv
Conv
Conv
Conv
FC
FC
FC
```

Won **ImageNet 2012**.

---

## 12.3 VGGNet (2014)

Key idea: use **only 3×3 convolutions**.

Architecture example (VGG16):

```
Conv
Conv
Pool
Conv
Conv
Pool
Conv
Conv
Conv
Pool
Conv
Conv
Conv
Pool
FC
FC
FC
```

Advantages:

* simple architecture
* deeper network

Problem:

```
138 million parameters
```

Very heavy.

---

## 12.4 GoogLeNet / Inception (2014)

Key idea: use **multiple convolution sizes in parallel**.

Inception block:

```
1×1 conv
3×3 conv
5×5 conv
pooling
```

All combined together.

Benefits:

* efficient
* fewer parameters

---

## 12.5 ResNet (2015)

Major breakthrough.

Key idea: **Skip Connections (Residual Learning)**.

Instead of learning:

```
H(x)
```

Learn:

```
F(x) = H(x) − x
```

So output becomes:

```
F(x) + x
```

Why?

Deep networks suffer from **vanishing gradients**.

ResNet solves this.

Example depths:

* ResNet34
* ResNet50
* ResNet101
* ResNet152

---

## 12.6 DenseNet

Idea: each layer connects to **every previous layer**.

```
Layer1 → Layer2
Layer1 → Layer3
Layer2 → Layer3
Layer1 → Layer4
Layer2 → Layer4
Layer3 → Layer4
```

Advantages:

* better gradient flow
* feature reuse

---

## 12.7 MobileNet

Designed for:

* mobile devices
* embedded systems

Key idea: **Depthwise Separable Convolution**

Instead of:

```
normal convolution
```

Split into:

```
depthwise convolution
+
pointwise convolution (1×1)
```

Result:

```
~9x fewer computations
```

---

# 13. Modern CNN Architecture Pattern

Most modern CNNs follow:

```
Input
↓
Conv
↓
BatchNorm
↓
ReLU
↓
Conv
↓
BatchNorm
↓
ReLU
↓
Pooling
↓
Repeat blocks
↓
Global Average Pooling
↓
Fully Connected
↓
Softmax
```

---

# 14. Conceptual CNN Architecture Summary

```
Image
 ↓
[Conv → ReLU] × N
 ↓
Pooling
 ↓
[Conv → ReLU] × N
 ↓
Pooling
 ↓
Flatten
 ↓
Fully Connected
 ↓
Softmax
```

---

# 15. Intuition of CNN Layers

| Layer   | What it detects  |
| ------- | ---------------- |
| Layer 1 | edges            |
| Layer 2 | corners          |
| Layer 3 | textures         |
| Layer 4 | parts of objects |
| Layer 5 | objects          |

This hierarchical learning is why CNNs work extremely well for vision.

---

# 16. Applications of CNN

CNNs are used in:

* image classification
* object detection
* medical imaging
* face recognition
* self-driving cars
* satellite analysis
* video understanding

---

# Key Concept

CNN architecture can be divided into **two phases**:

```
Feature Extractor
    (Convolution layers)

+
Classifier
    (Fully connected layers)
```

Modern networks often remove FC layers and use:

```
Global Average Pooling
```

to reduce parameters.
