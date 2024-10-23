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

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Parameters:
            X (ndarray): Feature dataset.
            y (ndarray): Target labels.
        """
        self.trees = []
        for i in range(self.n_estimators):
            # Create a bootstrap sample of the data
            X_sample, y_sample = self._bootstrap_sample(X, y)
            # Initialize a Decision Tree with specified parameters
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            # Randomly select features for splitting
            tree.feature_indices = self._get_feature_indices(X.shape[1])
            # Fit the tree on the bootstrap sample
            tree.fit(X_sample[:, tree.feature_indices], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
            X (ndarray): Input samples.

        Returns:
            ndarray: Predicted class labels.
        """
        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X[:, tree.feature_indices]) for tree in self.trees])
        # Majority vote
        y_pred = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(y_pred)
