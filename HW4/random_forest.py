import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.tree import ExtraTreeClassifier


class RandomForest(object):
    def __init__(self, n_estimators, max_depth, max_features, random_seed=None):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.max_features = max_features
        self.random_seed = random_seed
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [
            ExtraTreeClassifier(max_depth=self.max_depth, criterion="entropy")
            for i in range(self.n_estimators)
        ]
        self.alphas = (
            []
        )  # Importance values for adaptive boosting extra credit implementation

    def _bootstrapping(self, num_training, num_features, random_seed=None):
        """
        TODO:
        - Set random seed if it is inputted
        - Randomly select a sample dataset of size num_training with replacement from the original dataset.
        - Randomly select certain number of features (num_features denotes the total number of features in X,
          max_features denotes the percentage of features that are used to fit each decision tree) without replacement from the total number of features.

        Args:
        - num_training: number of data points in the bootstrapped dataset.
        - num_features: number of features in the original dataset.

        Return:
        - row_idx: the row indices corresponding to the row locations of the selected samples in the original dataset.
        - col_idx: the column indices corresponding to the column locations of the selected features in the original feature list.
        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        Hint 1: Please use np.random.choice. First get the row_idx first, and then second get the col_idx.
        Hint 2:  If you are getting a Test Failed: 'bool' object has no attribute 'any' error, please try flooring, or converting to an int, the number of columns needed for col_idx. Using np.ceil() can cause an autograder error.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    
        row_idx = np.random.choice(num_training, size=num_training, replace=True)
        num_features_to_use = int(num_features * self.max_features)

        col_idx = np.random.choice(num_features, size=num_features_to_use, replace=False)
        return row_idx, col_idx

    def bootstrapping(self, num_training, num_features):
        # helper function. You don't have to modify it
        # Initializing the bootstap datasets for each tree
        np.random.seed(self.random_seed)
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        TODO:
        Train decision trees using the bootstrapped datasets.
        Note that you need to use the row indices and column indices.
        X: NxD numpy array, where N is number
           of instances and D is the dimensionality of each
           instance
        y: 1D numpy array of size (N,), the predicted labels
        Returns:
            None. Calling this function should train the decision trees held in self.decision_trees
        """
        num_training = X.shape[0]
        num_features = X.shape[1]
        
        self.bootstrapping(num_training, num_features)
        
        for i in range(self.n_estimators):
            row_idx = self.bootstraps_row_indices[i]
            col_idx = self.feature_indices[i]
            
            X_bootstrap = X[row_idx][:, col_idx]
            y_bootstrap = y[row_idx]
            
            self.decision_trees[i].fit(X_bootstrap, y_bootstrap)

    def adaboost(self, X, y):
        """
        TODO:
        - Implement AdaBoost training by adjusting sample weights after each round of training a weak learner.
        - Begin by initializing equal weights for each training sample.
        - For each weak learner:
            - Train the learner on the current sample weights.
            - Get predictions for the training data using the learner's `predict()` method.
            - Calculate the weighted error of the sample and normalize.
            - Calculate `alpha`, the importance of the tree, using the formula from the notebook, and store it.
            - Update the weights of the samples using the formula from the notebook and normalize.

        Args:
        - X: NxD numpy array, feature set.
        - y: 1D numpy array of size (N,), labels.
        Returns:
            None. Trains the ensemble using AdaBoost's weighting mechanism.
        """
        N = X.shape[0]
        weights = np.ones(N) / N
        self.alphas = []
    
        for t in range(self.n_estimators):
            sample_indices = np.random.choice(N, size=N, replace=True, p=weights)
            X_sampled = X[sample_indices]
            y_sampled = y[sample_indices]
        
            self.decision_trees[t].fit(X_sampled, y_sampled)
            predictions = self.decision_trees[t].predict(X)
        
            # Calculate error 
            misclassified = (predictions != y)
            error = np.sum(weights * misclassified) / np.sum(weights)
            error = np.clip(error, 1e-10, 1 - 1e-10)
        
            # Calculate alpha 
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
        
            # Update & normalize weights
            weights = weights * np.exp(alpha * (misclassified * 2 - 1))
            weights = weights / np.sum(weights)
    
        return


    def OOB_score(self, X, y):
        # Helper function. You don't have to modify it.
        # This function computes the accuracy of the random forest model predicting y given x.
        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(
                        self.decision_trees[t].predict(
                            np.reshape(X[i][self.feature_indices[t]], (1, -1))
                        )[0]
                    )
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)

    def predict(self, X):
        N = X.shape[0]
        y = np.zeros((N, 7))
        for t in range(self.n_estimators):
            X_curr = X[:, self.feature_indices[t]]
            y += self.decision_trees[t].predict_proba(X_curr)
        pred = np.argmax(y, axis=1)
        return pred

    def predict_adaboost(self, X):
        # Helper method. You don't have to modify it.
        # This function makes predictions using AdaBoost ensemble by aggregating weighted votes.
        N = X.shape[0]
        weighted_votes = np.zeros((N, 7))

        for alpha, tree in zip(self.alphas, self.decision_trees[: len(self.alphas)]):
            pred = tree.predict(X)
            for i in range(N):
                class_index = int(pred[i])
                weighted_votes[i, class_index] += alpha

        return np.argmax(weighted_votes, axis=1)

    def plot_feature_importance(self, data_train):
        """
        TODO:
        -Display a bar plot showing the feature importance of every feature in
        one decision tree of your choice from the tuned random_forest from Q3.2.
        Args:
            data_train: This is the orginal data train Dataframe containg data AND labels.
                Hint: you can access labels with data_train.columns
        Returns:
            None. Calling this function should simply display the aforementioned feature importance bar chart
        """
        tree = self.decision_trees[0]
        importances = tree.feature_importances_
        
        if hasattr(data_train, 'columns'):
            feature_names = data_train.columns.tolist()[:-1]  
        else: 
            feature_names = [f"Feature {i}" for i in range(data_train.shape[1])]

        tree_features = self.feature_indices[0]
        feature_importance_pairs = []

        for i, importance in enumerate(importances):
            if importance > 0:  
                original_feature_idx = tree_features[i]
                if original_feature_idx < len(feature_names):
                    feature_name = feature_names[original_feature_idx]
                    feature_importance_pairs.append((feature_name, importance))
    
        feature_importance_pairs.sort(key=lambda x: x[1])
        sorted_names = [pair[0] for pair in feature_importance_pairs]
        sorted_importances = [pair[1] for pair in feature_importance_pairs]
    
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.barh(range(len(sorted_names)), sorted_importances, align='center')
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

    def hyperparameter_grid_search(
        self, n_estimators_range, max_depth_range, max_features_range
    ):
        """
        Hyperparameter tuning function with customizable ranges.

        Args:
            n_estimators_range (tuple): A tuple (start, stop, step) for n_estimators.
            max_depth_range (tuple): A tuple (start, stop, step) for max_depth.
            max_features_range (tuple): A tuple (start, stop, step) for max_features.

        Note: The range for a hyperparameter is inclusive of start and stop values. i.e.
        (start=8, stop=10, step=1) implies the range of values that that hyperparameter can take on
        is [8, 9, 10].

        Returns:
            A list of tuples, where each tuple represents a unique combination of hyperparameters.

        Example:
            hyperparameter_grid_search((8, 10, 1), (8, 10, 1), (0.7, 1.0, 0.1))

            Output:
            [
                (8, 8, 0.7),
                (8, 8, 0.8),
                ...
                (10, 10, 1.0)
            ]
        """
        n_estimators_values = list(range(n_estimators_range[0], n_estimators_range[1] + 1, n_estimators_range[2]))
        max_depth_values = list(range(max_depth_range[0], max_depth_range[1] + 1, max_depth_range[2]))
    
        max_features_values = []
        current_value = max_features_range[0]
        while current_value <= max_features_range[1] + 1e-10:  
            max_features_values.append(round(current_value, 1)) 
            current_value += max_features_range[2]
    
        combinations = []
        for n_est in n_estimators_values:
            for depth in max_depth_values:
                for features in max_features_values:
                    combinations.append((n_est, depth, features))
    
        return combinations
