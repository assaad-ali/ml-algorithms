import numpy as np
from ..base.base_model import BaseModel

class DecisionTreeNode:
    """
    A node in the decision tree.
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        """
        Initializes the node.
        """
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold          # Threshold value for the split
        self.left = left                    # Left child node
        self.right = right                  # Right child node
        self.value = value                  # Value at leaf node
        self.feature_indices = None         # Feature indices for splitting

    def is_leaf_node(self):
        """
        Checks if the node is a leaf node.
        """
        return self.value is not None


class DecisionTreeClassifier(BaseModel):
    """
    Decision Tree Classifier implementing CART algorithm.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initializes the Decision Tree Classifier.
        """
        self.max_depth = max_depth                  # Maximum depth of the tree
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split a node
        self.root = None                            # Root node of the tree

    def fit(self, X, y):
        """
        Builds the decision tree classifier.
        """
        self.n_classes_ = len(set(y))  # Number of classes
        self.n_features_ = X.shape[1]  # Number of features
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predicts class labels for samples in X.
        """
        return np.array([self._predict(inputs) for inputs in X])

    def _gini(self, y):
        """
        Calculates the Gini Impurity for labels y.
        """
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)

        node = DecisionTreeNode(value=predicted_class)

        # Check stopping criteria
        if depth < self.max_depth and len(y) >= self.min_samples_split and self._gini(y) > 0:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node = DecisionTreeNode(feature_index=idx, threshold=thr)
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        """
        Finds the best feature and threshold to split on.

        Parameters:
            X (ndarray): Feature dataset.
            y (ndarray): Target labels.

        Returns:
            best_idx (int): Index of the best feature to split on.
            best_thr (float): Threshold value to split the feature.
        """
        m, n = X.shape
        if m <= 1:
            return None, None

        # Initialize variables to track the best split
        best_gini = 1.0
        best_idx, best_thr = None, None

        # Determine which features to consider
        features = self.feature_indices if self.feature_indices is not None else range(n)

        # Loop over selected features
        for idx in features:
            # Sort data along the feature axis
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.n_classes_
            num_right = num_samples_per_class = [np.sum(y == c) for c in range(self.n_classes_)]

            for i in range(1, m):  # Possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_) if i != 0)
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_) if (m - i) != 0)

                # Weighted average of the impurity
                gini = (i * gini_left + (m - i) * gini_right) / m

                # Skip if the threshold is the same as the previous one
                if thresholds[i] == thresholds[i - 1]:
                    continue 

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # Midpoint

        return best_idx, best_thr


    def _predict(self, inputs):
        """
        Predicts a single sample.
        """
        node = self.root
        while not node.is_leaf_node():
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
