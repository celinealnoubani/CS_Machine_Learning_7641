# HW2: Clustering Algorithms

## Course: CS 7641 - Machine Learning (Georgia Tech)

## Overview
This assignment implements three fundamental unsupervised learning algorithms for clustering: K-Means, Gaussian Mixture Models (GMM), and DBSCAN. The implementations are applied to real-world problems including fraud detection and image compression.

## Topics Covered
- **K-Means Clustering** - Centroid-based partitioning with multiple initialization strategies
- **Expectation-Maximization (EM) Algorithm** - Probabilistic framework for parameter estimation
- **Gaussian Mixture Models (GMM)** - Soft clustering with probabilistic assignments
- **DBSCAN** - Density-based spatial clustering for non-convex shapes

## Implementation Details

### 1. K-Means Clustering (`kmeans.py`)

#### Core Components
| Function | Description |
|----------|-------------|
| `pairwise_dist(x, y)` | Vectorized Euclidean distance using X² + Y² - 2XY formula |
| `init_centers()` | Random initialization by sampling from dataset |
| `kmpp_init()` | K-Means++ initialization for better convergence |
| `update_assignment()` | Assigns points to nearest centroid |
| `update_centers()` | Recomputes centroids from cluster members |
| `get_loss()` | Sum of squared distances (inertia) |
| `train()` | Full training loop with convergence check |
| `fowlkes_mallow()` | Clustering quality metric vs ground truth |

#### K-Means++ Algorithm
Implemented intelligent center initialization:
1. Sample 1% of data uniformly at random
2. Select first center randomly
3. Choose subsequent centers based on maximum squared distance
4. Repeat until K centers are selected

### 2. Gaussian Mixture Models (`gmm.py`)

#### Helper Functions
| Function | Description |
|----------|-------------|
| `softmax(logit)` | Numerically stable softmax with max subtraction |
| `logsumexp(logit)` | Log-sum-exp trick for numerical stability |
| `normalPDF()` | Diagonal covariance Gaussian PDF (undergrad) |
| `multinormalPDF()` | Full covariance multivariate Gaussian PDF (grad) |

#### EM Algorithm Implementation
- **E-Step (`_E_step`)**: Compute responsibilities (posterior probabilities)
  ```
  τ(zₖ) = π_k * N(x|μ_k, Σ_k) / Σⱼ π_j * N(x|μ_j, Σ_j)
  ```
- **M-Step (`_M_step`)**: Update parameters
  - π_k = N_k / N (mixing coefficients)
  - μ_k = Σ τ(zₖ)xₙ / N_k (means)
  - Σ_k = weighted covariance matrix

#### Applications
- **Image Compression**: Cluster pixels by RGB values, replace with cluster means
- **Density Estimation**: Fit GMM to 2D data and sample via rejection sampling

### 3. DBSCAN (`dbscan.py`)

#### Core Components
| Function | Description |
|----------|-------------|
| `regionQuery(pointIndex)` | Find all points within ε-neighborhood |
| `expandCluster()` | Recursively expand cluster from core point |
| `fit()` | Main algorithm - identifies core, border, and noise points |

**Key Parameters**:
- `eps`: Maximum radius of neighborhood
- `minPts`: Minimum points required for dense region

**Advantage over K-Means**: Successfully clusters non-convex shapes like concentric circles

## Applications Demonstrated
- **Fraud Detection**: Clustering credit card transactions to identify anomalies
- **Image Compression**: Reducing color palette using GMM clustering
- **Density-based Clustering**: Handling arbitrary cluster shapes with DBSCAN

## Files
- `kmeans.py` - K-Means implementation
- `gmm.py` - GMM with EM algorithm
- `dbscan.py` - DBSCAN implementation
- `HW2.ipynb` - Jupyter notebook with experiments

## Key Concepts Demonstrated
- Vectorized distance computations without loops
- Numerical stability techniques (log-sum-exp, softmax stabilization)
- Convergence criteria for iterative algorithms
- Handling edge cases (empty clusters, singular covariance matrices)
- Comparison of hard vs soft clustering approaches

## Skills Gained
- Implementation of unsupervised learning algorithms from scratch
- Understanding EM algorithm mechanics
- Practical experience with clustering applications
- Handling numerical issues in probabilistic models
