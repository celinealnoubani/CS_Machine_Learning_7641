# HW4: Neural Networks & Ensemble Methods

## Course: CS 7641 - Machine Learning (Georgia Tech)

## Overview
This assignment implements deep learning fundamentals from scratch, including a multi-layer neural network with modern optimization techniques, convolutional neural networks using PyTorch, and ensemble methods with Random Forests. Also covers Support Vector Machines (SVM) theory.

## Topics Covered
- **Multi-Layer Neural Networks** - Forward/backward propagation from scratch
- **Optimization Algorithms** - Gradient descent, mini-batch GD, and Adam optimizer
- **Regularization** - Dropout for preventing overfitting
- **Convolutional Neural Networks** - Image classification with PyTorch
- **Random Forests** - Ensemble learning with decision trees
- **Support Vector Machines** - Kernel methods and margin maximization

## Implementation Details

### 1. Two-Layer Neural Network (`NN.py`)

#### Architecture
```
Input (8) → Hidden1 (15) → Hidden2 (7) → Output (3)
         SiLU          SiLU         Softmax
```

#### Activation Functions
| Function | Formula | Description |
|----------|---------|-------------|
| `silu(u)` | u × σ(u) | Sigmoid Linear Unit (Swish) |
| `derivative_silu(x)` | σ(x)(1 + x(1-σ(x))) | SiLU gradient for backprop |
| `softmax(u)` | exp(u)/Σexp(u) | Output probabilities (numerically stable) |

#### Core Methods
| Method | Description |
|--------|-------------|
| `forward(x, use_dropout)` | Forward pass through all layers |
| `cross_entropy_loss(y, y_hat)` | Multi-class cross-entropy: -Σ y log(ŷ) |
| `compute_gradients(y, yh)` | Backpropagation to compute all gradients |
| `update_weights(dLoss)` | Apply gradients to update parameters |
| `backward(y, yh)` | Complete backward pass |

#### Regularization: Dropout
```python
def _dropout(u, prob):
    dropout_mask = np.random.choice([0, 1], size=u.shape, p=[prob, 1-prob])
    u_after_dropout = (u * dropout_mask) / (1 - prob)  # Inverted dropout
    return u_after_dropout, dropout_mask
```
- Applied only to first hidden layer
- Inverted dropout scales activations during training

#### Optimization Methods

**Gradient Descent**
```python
def gradient_descent(self, x, y, iter=60000):
    # Full batch gradient descent
    for i in range(iter):
        yh = self.forward(x, use_dropout)
        loss = self.cross_entropy_loss(y, yh)
        self.backward(y, yh)
```

**Mini-batch Gradient Descent**
- Wraparound batching for consistent batch sizes
- Reduces variance in gradient estimates
- Example: [1,2,3,4,5,6,7,8,9] with batch_size=6 → [1-6], [7-9,1-3], [4-9]...

**Adam Optimizer**
```python
# First moment (momentum): m = β₁m + (1-β₁)g
# Second moment (RMSprop): v = β₂v + (1-β₂)g²
# Bias correction: m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ)
# Update: θ = θ - α × m̂/(√v̂ + ε)
```
Parameters: β₁=0.9, β₂=0.999, ε=1e-8

### 2. Convolutional Neural Networks (`cnn.py`, `cnn_trainer.py`)

#### CNN Architecture (PyTorch)
- Convolutional layers for feature extraction
- Pooling layers for spatial reduction
- Fully connected layers for classification

#### Data Augmentation (`cnn_image_transformations.py`)
Implemented image transformations to improve model generalization:
- Random horizontal/vertical flips
- Random rotations
- Color jittering
- Random cropping

#### Training Pipeline
- DataLoader with batching
- Cross-entropy loss
- SGD/Adam optimization
- Learning rate scheduling

### 3. Random Forest (`random_forest.py`)

#### Implementation
| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train ensemble of decision trees |
| `predict(X)` | Aggregate predictions via majority voting |
| `grid_search(...)` | Hyperparameter optimization |

#### Hyperparameter Tuning
- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split a node
- `max_features`: Features considered per split

#### Evaluation
- Confusion matrix analysis
- Grid search for optimal parameters
- Feature importance analysis

### 4. Support Vector Machines (Theory)

#### Key Concepts
- **Maximum Margin Classifier**: Find hyperplane maximizing distance to nearest points
- **Support Vectors**: Data points on the margin boundaries
- **Kernel Trick**: Transform data to higher dimensions for non-linear separation

#### Kernels Covered
- Linear kernel: K(x,y) = xᵀy
- Polynomial kernel: K(x,y) = (xᵀy + c)ᵈ
- RBF/Gaussian kernel: K(x,y) = exp(-γ||x-y||²)

## Applications Demonstrated

### California Housing Dataset
- Regression on housing prices
- Multi-class classification via binning
- Neural network training from scratch

### Image Classification
- CNN-based image recognition
- Data augmentation strategies
- Transfer learning concepts

### Ensemble Methods
- Random Forest for tabular data
- Hyperparameter tuning via grid search
- Model interpretation with feature importance

## Files
- `NN.py` - Neural network from scratch
- `cnn.py` - CNN architecture definition
- `cnn_trainer.py` - CNN training utilities
- `cnn_image_transformations.py` - Data augmentation
- `random_forest.py` - Random Forest implementation
- `HW4.ipynb` - Jupyter notebook with experiments

## Key Concepts Demonstrated
- Backpropagation algorithm derivation and implementation
- Modern optimization techniques (Adam)
- Regularization strategies (Dropout)
- Ensemble methods for improved generalization
- CNN architecture design for image data

## Skills Gained
- Building neural networks from scratch (no frameworks)
- Understanding gradient flow in deep networks
- Implementing modern optimizers
- Using PyTorch for deep learning
- Ensemble learning with Random Forests
- Hyperparameter tuning strategies
- Model evaluation and interpretation
