import numpy as np
from ml_algorithms.base.base_model import BaseModel

class SVM(BaseModel):
    """
    Support Vector Machine classifier using a linear kernel.

    Parameters:
        learning_rate (float): Learning rate for the optimization algorithm.
        lambda_param (float): Regularization parameter.
        n_iters (int): Number of iterations over the training set.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        """
        Train the SVM model.

        Parameters:
            X (ndarray): Training data features.
            y (ndarray): Training data labels.
        """
        n_samples, n_features = X.shape

        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent Optimization
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # No misclassification
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Misclassification
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
            X (ndarray): Input samples.

        Returns:
            ndarray: Predicted class labels.
        """
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
