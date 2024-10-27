import numpy as np
from ml_algorithms.tree.decision_tree_base import DecisionTreeBase

class DecisionTreeRegressor(DecisionTreeBase):
    """
    Decision Tree Regressor using the CART algorithm.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        super().__init__(max_depth, min_samples_split)

    def _variance(self, y):
        """
        Calculates the variance of y.
        """
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _best_split(self, X, y, n_samples, n_features):
        """
        Finds the best feature and threshold to split on.
        """
        best_mse = float('inf')
        best_feature_index = None
        best_threshold = None

        # Loop over features
        features = self.feature_indices if self.feature_indices is not None else range(n_features)

        for feature_index in features:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # Split the dataset
                indices_left = X[:, feature_index] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Calculate the MSE for left and right splits
                mse_left = self._variance(y_left)
                mse_right = self._variance(y_right)
                mse_total = (len(y_left) * mse_left + len(y_right) * mse_right) / n_samples

                if mse_total < best_mse:
                    best_mse = mse_total
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _calculate_leaf_value(self, y):
        """
        Calculates the mean of y.
        """
        return np.mean(y)
