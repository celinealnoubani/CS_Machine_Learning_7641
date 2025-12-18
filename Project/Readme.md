# METABRIC Breast Cancer Subtype Classification

## Course: CS 7641 - Machine Learning (Georgia Tech)

**Project Website**: [https://github.gatech.edu/pages/lramirez65/CS7641_ML_group3/index.html](https://github.gatech.edu/pages/lramirez65/CS7641_ML_group3/index.html)

## Overview

This project applies machine learning techniques to classify breast cancer molecular subtypes using the METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset. We implement and compare 5 different classification models with hyperparameter tuning and model explainability analysis using SHAP values.

## Problem Description

- **Task**: Multi-class classification of breast cancer molecular subtypes
- **Target Variable**: `pam50_+_claudin-low_subtype`
- **Classes**: 6 subtypes (after removing NC)
  - LumA (Luminal A)
  - LumB (Luminal B)
  - Her2 (HER2-enriched)
  - Basal (Basal-like)
  - claudin-low
  - Normal (Normal-like)
- **Features**: 50 gene expression features
- **Training Samples**: ~1,521
- **Test Samples**: 377

## Dataset

The METABRIC dataset contains gene expression profiles and clinical data from breast cancer patients. The PAM50 classification system along with Claudin-low subtyping provides clinically relevant molecular subtypes that inform treatment decisions.

### Class Distribution
| Subtype | Training Count |
|---------|---------------|
| LumA | 552 |
| LumB | 369 |
| Her2 | 169 |
| Basal | 162 |
| claudin-low | 160 |
| Normal | 109 |

---

## Pipeline

### 1. Preprocessing & EDA (`1_preprocessing_EDA.ipynb`)
- Load and explore raw METABRIC data
- Handle missing values and remove NC (Not Classified) samples
- Feature selection and normalization
- Train/test split generation
- Class distribution visualization

**Run this notebook first** to generate `training_dataset.csv` and `test_dataset.csv`.

---

## Model Implementations

### 2. Logistic Regression (`2_logisticRegression.ipynb`)

**Configuration:**
- Multinomial classification with L2 regularization
- Hyperparameter search: C, solver, penalty
- RandomizedSearchCV with 5-fold CV

**Results:**
| Metric | Value |
|--------|-------|
| Test Accuracy | 76% |
| Weighted AUROC | 0.9475 |
| Best CV AUC | 0.9525 |

**Best Parameters:** `solver='newton-cg', penalty='l2'`

---

### 3. Gradient Boosting Decision Trees (`3_GBDT.ipynb`)

**Configuration:**
- GradientBoostingClassifier with balanced sample weights
- Hyperparameter search: n_estimators, learning_rate, max_depth, subsample

**Results:**
| Metric | Value |
|--------|-------|
| Test Accuracy | 77% |
| Weighted AUROC | 0.9449 |
| Best CV AUC | 0.9519 |

**Best Parameters:** `n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.9`

---

### 4. Decision Tree (`4_DT.ipynb`)

**Configuration:**
- DecisionTreeClassifier with various splitting criteria
- Hyperparameter search: max_depth, min_samples_split, min_samples_leaf, criterion

**Results:**
| Metric | Value |
|--------|-------|
| Test Accuracy | 52% |
| Weighted AUROC | 0.8379 |
| Best CV AUC | 0.8473 |

**Note:** Single decision tree shows limited performance compared to ensemble methods.

---

### 5. Support Vector Machine (`5_SVM.ipynb`)

**Configuration:**
- SVC with probability estimates enabled
- Hyperparameter search: C, kernel, gamma, degree

**Results:**
| Metric | Value |
|--------|-------|
| Test Accuracy | 76% |
| Weighted AUROC | 0.9514 |
| Best CV AUC | 0.9545 |

**Best Parameters:** `kernel='rbf', C=100, gamma=0.001`

---

### 6. Random Forest (`6_RF.ipynb`)

**Configuration:**
- RandomForestClassifier with balanced class weights and OOB scoring
- Hyperparameter search: n_estimators, max_depth, min_samples_split, max_features

**Results:**
| Metric | Value |
|--------|-------|
| Test Accuracy | 77% |
| Weighted AUROC | 0.9474 |
| Best CV AUC | 0.9525 |

**Best Parameters:** `n_estimators=1000, max_depth=20, min_samples_split=5, min_samples_leaf=2, max_features='log2'`

---

## Model Comparison

| Model | Test Accuracy | Weighted AUROC | Best CV AUC |
|-------|--------------|----------------|-------------|
| **SVM (RBF)** | 76% | **0.9514** | **0.9545** |
| Random Forest | **77%** | 0.9474 | 0.9525 |
| Logistic Regression | 76% | 0.9475 | 0.9525 |
| GBDT | **77%** | 0.9449 | 0.9519 |
| Decision Tree | 52% | 0.8379 | 0.8473 |

**Key Findings:**
- **Best model selection based on weighted test AUROC** (not accuracy), as AUROC better captures performance across imbalanced classes
- **SVM is the best model** with highest AUROC (0.9514), followed closely by Logistic Regression (0.9475)
- **Data is linearly separable** as supported by UMAP visualization, explaining why simpler models perform well
- **Per-class F1 analysis**: LR wins on 3 classes (Basal, Claudin-low, Normal); RF wins LumB; GBDT wins Her2; SVM sits in the middle across classes
- **LumA & LumB confusion**: Both SVM and LR show high misclassification between these similar subtypes; LumA samples often misclassified as Normal
- **Decision Tree is the weakest model**: Shallow depth leads to high bias and underfitting, using only a handful of features per split
- **Ensemble advantage**: RF and GBDT aggregate hundreds of shallow trees, allowing feature diversity and noise smoothing that single DT lacks

---

## Model Explainability (SHAP Analysis)

Each model notebook includes SHAP (SHapley Additive exPlanations) analysis to identify the most important gene features for each breast cancer subtype.

### Analysis Components:
- **SHAP Summary Plots**: Global feature importance across all classes
- **Per-Class SHAP Values**: Top 10 most informative genes for each subtype
- **Marker Gene Tables**: Coefficients/SHAP values of top genes across all subtypes
- **Expression Boxplots**: Gene expression patterns across subtypes

### Clinical Relevance:
- **Basal & Claudin-low**: Aggressive subtypes with distinct gene expression signatures
- Identified marker genes can inform targeted therapy decisions
- SHAP analysis provides biological interpretability for model predictions

---

## Technical Implementation

### Common Pipeline:
```python
# 1. Load preprocessed data
train_df = pd.read_csv('training_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

# 2. Handle class imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 3. Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# 4. Hyperparameter tuning
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    scoring={'accuracy', 'f1_weighted', 'roc_auc_ovr'},
    refit='roc_auc_ovr',
    cv=5
)

# 5. Evaluate
evaluate_model(best_model, X_test, y_test, le.classes_)
```

### Evaluation Metrics:
- **Accuracy**: Overall classification accuracy
- **Weighted F1-Score**: Accounts for class imbalance
- **Weighted AUROC**: One-vs-Rest ROC AUC, weighted by class support
- **Confusion Matrix**: Per-class prediction analysis
- **Per-Class ROC Curves**: AUC for each subtype

---

## Repository Structure

```
Project/
├── Notebooks/
│   ├── 1_preprocessing_EDA (1).ipynb    # Data preprocessing & exploration
│   ├── 2_logisticRegression.ipynb       # Logistic Regression model
│   ├── 3_GBDT.ipynb                     # Gradient Boosting model
│   ├── 4_DT.ipynb                       # Decision Tree model
│   ├── 5_SVM.ipynb                      # Support Vector Machine model
│   └── 6_RF.ipynb                       # Random Forest model
├── Data/
│   ├── dataset.csv                      # Original METABRIC data
│   ├── training_dataset.csv             # Preprocessed training set
│   └── test_dataset.csv                 # Preprocessed test set
├── Docs/
│   ├── Final_Presentation (1).pdf       # Project presentation slides
│   └── index.html                       # Project website
└── README.md
```

---

## Technologies Used

- **Python 3.11**
- **Scikit-learn** - ML models and evaluation
- **SHAP** - Model explainability
- **Pandas/NumPy** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **Google Colab** - Development environment

---

## Key Concepts Demonstrated

- Multi-class classification with imbalanced data
- Hyperparameter optimization with cross-validation
- Model comparison and selection
- Feature importance and model interpretability
- Biological/clinical relevance of ML predictions
- Ensemble methods vs. single classifiers

---

## References

- METABRIC Dataset: Curtis et al., Nature 2012
- PAM50 Classification: Parker et al., JCO 2009
- Claudin-low Subtype: Prat et al., Breast Cancer Research 2010
- SHAP: Lundberg & Lee, NeurIPS 2017
