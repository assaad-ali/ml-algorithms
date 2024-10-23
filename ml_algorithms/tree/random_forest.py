import numpy as np
from ml_algorithms.base.base_model import BaseModel
from ml_algorithms.tree.decision_tree import DecisionTreeClassifier
from collections import Counter

class RandomForestClassifier(BaseModel):
    """
    Random Forest Classifier using an ensemble of Decision Trees.

    Parameters:
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        max_features (int or float or str): Number of features to consider when looking for the best split.
        bootstrap (bool): Whether to use bootstrap samples when building trees.
        random_state (int): Seed used by the random number generator.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []  # List to store the individual Decision Trees

        # Set the random seed for reproducibility
        np.random.seed(self.random_state)
