from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract Base Class for all machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the data X and target y.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict using the fitted model on data X.
        """
        pass
