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
