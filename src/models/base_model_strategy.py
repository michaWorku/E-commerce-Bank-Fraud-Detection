from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModelStrategy(ABC):
    """
    Abstract Base Class for various machine learning model strategies.
    Defines the interface for training and prediction.
    """
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the machine learning model.

        Parameters:
        X (pd.DataFrame): Training features.
        y (pd.Series): Target variable for training.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Parameters:
        X (pd.DataFrame): Features for prediction.

        Returns:
        np.ndarray: Array of predictions.
        """
        pass

    @abstractmethod
    def get_model(self):
        """
        Returns the trained model object.
        """
        return self.model

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the model strategy.
        """
        pass
