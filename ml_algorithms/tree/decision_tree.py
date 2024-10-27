import numpy as np
from ml_algorithms.tree.decision_tree_base import DecisionTreeBase

class DecisionTreeClassifier(DecisionTreeBase):
    """
    Decision Tree Classifier implementing CART algorithm.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        super().__init__(max_depth, min_samples_split)
        self.n_classes_ = None

    def fit(self, X, y):
        """
        Builds the decision tree classifier.
        """
        self.n_classes_ = len(set(y))
        super().fit(X, y)

    def _gini(self, y):
        """
        Calculates the Gini Impurity for labels y.
        """
        m = len(y)
        class_counts = np.bincount(y)
        return 1.0 - np.sum((class_counts / m) ** 2)

    def _entropy(self, y):
        """
        Calculates the Entropy for labels y.
        """
        m = len(y)
        class_counts = np.bincount(y)
        probabilities = class_counts / m
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _best_split(self, X, y, n_samples, n_features):
        """
        Finds the best feature and threshold to split on.
        """
        best_impurity = float('inf')
        best_feature_index = None
        best_threshold = None

        # Loop over features
        features = self.feature_indices if self.feature_indices is not None else range(n_features)

        for feature_index in features:
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # Split the dataset
                indices_left = X_column < threshold
                indices_right = ~indices_left

                if len(y[indices_left]) == 0 or len(y[indices_right]) == 0:
                    continue

                impurity = self._information_gain(y, y[indices_left], y[indices_right])

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _information_gain(self, parent, left_child, right_child):
        """
        Calculates the information gain from a split.
        """
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)

        gain = self._gini(parent) - (weight_left * self._gini(left_child) + weight_right * self._gini(right_child))
        return gain

    def _calculate_leaf_value(self, y):
        """
        Calculates the most common class label.
        """
        return np.argmax(np.bincount(y))
