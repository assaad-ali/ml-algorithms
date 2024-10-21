import numpy as np
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

