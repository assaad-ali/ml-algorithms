import numpy as np
from abc import ABC, abstractmethod
from ml_algorithms.base.base_model import BaseModel

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

    def is_leaf_node(self):
        """
        Checks if the node is a leaf node.
        """
        return self.value is not None

class DecisionTreeBase(BaseModel, ABC):
    """
    Abstract base class for Decision Trees.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initializes the Decision Tree.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features_ = None
        self.feature_indices = None  # For feature subsetting (e.g., in Random Forests)

    def fit(self, X, y):
        """
        Builds the decision tree.
        """
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predicts outputs for samples in X.
        """
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.
        """
        n_samples, n_features = X.shape

        # Check stopping criteria
        if depth >= self.max_depth if self.max_depth is not None else False \
                or n_samples < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(y)
            return DecisionTreeNode(value=leaf_value)

        # Find the best split
        feature_index, threshold = self._best_split(X, y, n_samples, n_features)
        if feature_index is None:
            leaf_value = self._calculate_leaf_value(y)
            return DecisionTreeNode(value=leaf_value)

        # Split the dataset
        indices_left = X[:, feature_index] < threshold
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        # Build the child nodes recursively
        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)
        return DecisionTreeNode(feature_index=feature_index, threshold=threshold,
                                left=left_child, right=right_child)

    def _predict(self, inputs):
        """
        Predicts output for a single sample.
        """
        node = self.root
        while not node.is_leaf_node():
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    @abstractmethod
    def _best_split(self, X, y, n_samples, n_features):
        """
        Finds the best feature and threshold to split on.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _calculate_leaf_value(self, y):
        """
        Calculates the value to be assigned at a leaf node.
        Must be implemented by subclasses.
        """
        pass
