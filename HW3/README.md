# HW3: Regression, Classification & Dimensionality Reduction

## Course: CS 7641 - Machine Learning (Georgia Tech)

## Overview
This assignment covers supervised learning fundamentals including regression models, logistic regression for classification, and dimensionality reduction techniques. Implementations emphasize understanding of optimization algorithms, regularization, and feature selection.

## Topics Covered
- **Principal Component Analysis (PCA)** - Dimensionality reduction via SVD
- **Linear & Polynomial Regression** - Closed-form and gradient-based solutions
- **Ridge Regression** - L2 regularization to prevent overfitting
- **Logistic Regression** - Binary classification with gradient descent
- **Feature Selection** - Information-theoretic feature reduction

## Implementation Details

### 1. PCA Implementation (`pca.py`)

#### Core Methods
| Method | Description |
|--------|-------------|
| `fit(X)` | Compute SVD on centered data to get U, S, V matrices |
| `transform(data, K)` | Project data onto top K principal components |
| `transform_rv(data, retained_variance)` | Select K to retain specified variance ratio |
| `visualize(X, y)` | 2D/3D scatter plots of PCA-reduced data |

#### Image Compression Application
- Used SVD to compress grayscale and color images
- Analyzed reconstruction quality vs number of components retained
- Demonstrated trade-off between compression ratio and image fidelity

### 2. Regression (`regression.py`)

#### Linear Regression Methods
| Method | Description |
|--------|-------------|
| `construct_polynomial_feats(x, degree)` | Generate polynomial features up to specified degree |
| `linear_fit_closed(X, y)` | Closed-form solution: w = (X^T X)^(-1) X^T y |
| `linear_fit_GD(X, y, epochs, lr)` | Gradient descent optimization |
| `linear_fit_SGD(X, y, epochs, lr)` | Stochastic gradient descent (one sample at a time) |

#### Ridge Regression Methods
| Method | Description |
|--------|-------------|
| `ridge_fit_closed(X, y, λ)` | Closed-form: w = (X^T X + λI)^(-1) X^T y |
| `ridge_fit_GD(X, y, λ, epochs, lr)` | Gradient descent with L2 penalty |
| `ridge_fit_SGD(X, y, λ, epochs, lr)` | SGD with regularization |
| `ridge_cross_validation(X, y, kfold, λ)` | K-fold CV for model evaluation |
| `hyperparameter_search(X, y, λ_list, kfold)` | Find optimal λ via cross-validation |

#### Key Implementation Details
- Bias term excluded from regularization penalty
- RMSE (Root Mean Square Error) as loss metric
- Polynomial feature expansion for non-linear relationships

### 3. Logistic Regression (`logistic_regression.py`)

#### Core Components
| Method | Description |
|--------|-------------|
| `sigmoid(s)` | Numerically stable sigmoid with clipping |
| `bias_augment(x)` | Prepend column of 1s for bias term |
| `predict_probs(x_aug, θ)` | Compute P(y=1\|x) = σ(θᵀx) |
| `predict_labels(h_x, threshold)` | Convert probabilities to binary labels |
| `loss(y, h_x)` | Binary cross-entropy loss |
| `gradient(x_aug, y, h_x)` | Gradient of loss w.r.t. parameters |
| `accuracy(y, y_hat)` | Classification accuracy metric |
| `fit(...)` | Training loop with gradient descent |

#### Training Features
- Validation loss/accuracy tracking every 100 epochs
- Threshold tuning for optimal classification boundary
- Loss and accuracy visualization plots

#### Hyperparameter Tuning
```python
hyperparameter_tuning(model, x_test, y_test, theta, thresholds)
```
Finds optimal classification threshold by evaluating accuracy across threshold values.

### 4. Feature Selection (`feature_reduction.py`)

Implemented information-theoretic feature selection methods:
- Mutual information computation
- Feature ranking by relevance
- Redundancy reduction in feature subsets

## Applications Demonstrated

### Image Compression with SVD
- Decompose image matrix using SVD
- Reconstruct using top K singular values
- Analyze compression vs quality trade-off

### Polynomial Regression
- Fit curves to non-linear data
- Compare different polynomial degrees
- Visualize underfitting vs overfitting

### News Sentiment Classification
- Binary classification of news headlines
- Feature extraction from text data
- Logistic regression for sentiment prediction

## Files
- `pca.py` - PCA implementation with SVD
- `regression.py` - Linear and Ridge regression
- `logistic_regression.py` - Logistic regression classifier
- `feature_reduction.py` - Feature selection methods
- `imgcompression.py` - Image compression utilities
- `HW3.ipynb` - Jupyter notebook with experiments

## Key Concepts Demonstrated
- Closed-form vs iterative optimization methods
- Regularization to control model complexity
- Cross-validation for hyperparameter selection
- Dimensionality reduction for visualization and compression
- Binary classification with probabilistic outputs

## Skills Gained
- Implementing ML algorithms from mathematical formulations
- Understanding gradient-based optimization (GD, SGD)
- Practical experience with regularization techniques
- Model selection via cross-validation
- Feature engineering and selection strategies
