# HW1: Programming Warm-Up

## Course: CS 7641 - Machine Learning (Georgia Tech)

## Overview
This assignment serves as a warm-up to familiarize with the programming environment and NumPy fundamentals used throughout the course. It focuses on efficient vectorized operations and array broadcasting techniques essential for implementing machine learning algorithms.

## Topics Covered
- **Python Environment Setup** - Configuring Conda environment with required packages
- **NumPy Fundamentals** - Core array operations without loops
- **Array Broadcasting** - Efficient distance computations using vectorization

## Implementation Details

### 1. NumPy Basics (`warmup.py`)
Implemented several NumPy one-liner functions without using loops:

| Function | Description |
|----------|-------------|
| `indices_of_k(arr, k)` | Returns indices where array values equal k using `np.where` |
| `argmax_1d(arr)` | Returns index of maximum value using `np.argmax` |
| `mean_rows(arr)` | Computes mean of each row using `np.mean` with axis parameter |
| `sum_squares(arr)` | Computes sum of squares per row, maintaining 2D shape with `keepdims` |

### 2. Vectorized Manhattan Distance (`fast_manhattan`)
Implemented a highly optimized pairwise Manhattan distance calculation using broadcasting:

```python
def fast_manhattan(x, y):
    return np.sum(np.abs(x[:, None, :] - y[None, :, :]), axis=2)
```

**Performance**: ~100x faster than the naive loop-based implementation (685 microseconds vs 98.8 milliseconds for 100x100 points)

## Key Concepts Demonstrated
- **Vectorization**: Replacing explicit Python loops with NumPy operations for performance
- **Broadcasting**: Leveraging NumPy's broadcasting rules to compute pairwise operations efficiently
- **Memory Efficiency**: Understanding time-space tradeoffs in vectorized computations

## Files
- `warmup.py` - NumPy implementations
- `HW1 (1).ipynb` - Jupyter notebook with exercises and tests
- `env.pkl` - Environment configuration

## Skills Gained
- NumPy array manipulation and indexing
- Vectorized computation techniques
- Understanding of broadcasting mechanics
- Performance optimization in scientific Python
