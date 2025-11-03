import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose

        Hint: np.linalg.svd by default returns the transpose of V
              Make sure you remember to first center your data by subtracting the mean of each feature.

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        self.U, self.S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.V = Vt
        

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        by projecting onto the principal components.
        Utilize class members that were previously set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
        """
        data_centered = data - self.mean
        X_new = data_centered @ self.V[:K, :].T
        
        return X_new


    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
        """
        data_centered = data - self.mean
        total_variance = np.sum(self.S**2)
        cumulative_variance_ratio = np.cumsum(self.S**2) / total_variance
        
        K = np.argmax(cumulative_variance_ratio >= retained_variance) + 1
        
        X_new = data_centered @ self.V[:K, :].T
        return X_new
   

    def get_V(self) -> np.ndarray:
        """
        Getter function for value of V
        """
        return self.V


    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) -> None:
        """
        You have to plot three different scatterplots (2D and 3D for strongest two features and 2D for two random features) for this function.
        For plotting the 2D scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later random) features.
        You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using matplotlib.

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels

        Return: None
        """
        self.fit(X)
        
        # 2D scatter plot of strongest two features
        X_pca_2d = self.transform(X, K=2)
        unique_labels = np.unique(y)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(unique_labels):
            plt.scatter(X_pca_2d[y == label, 0], X_pca_2d[y == label, 1], color=colors[i], alpha=0.7)
        
        plt.title(f'{fig_title} - 2D PCA Reduction')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
        
        # 3D scatter plot of strongest two features
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    
        for i, label in enumerate(unique_labels):
            ax.scatter(X_pca_2d[y == label, 0], X_pca_2d[y == label, 1], 
            np.zeros_like(X_pca_2d[y == label, 0]),
            color=colors[i], alpha=0.7)
    
        ax.set_title(f'{fig_title} - 3D PCA Reduction')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.show()

        # 2D plot of two random features
        n_features = X.shape[1]
        if n_features >= 2:
            random_indices = np.random.choice(n_features, 2, replace=False)
            
            plt.figure(figsize=(8, 6))
            for i, label in enumerate(unique_labels):
                plt.scatter(X[y == label, random_indices[0]], X[y == label, random_indices[1]], 
                color=colors[i], alpha=0.7)
            
            plt.title(f'{fig_title} - Randomly Selected Features')
            plt.xlabel(f'Feature {random_indices[0]}')
            plt.ylabel(f'Feature {random_indices[1]}')
            plt.show()
        
    
