import pandas as pd
import numpy as np
from typing import Dict, Any

from pathlib import Path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.base_model_strategy import BaseModelStrategy


class ModelTrainer:
    """
    Context class for training and predicting with various machine learning models
    using a strategy pattern.
    """
    def __init__(self, strategy: BaseModelStrategy):
        """
        Initializes the ModelTrainer with a specific modeling strategy.

        Parameters:
        strategy (BaseModelStrategy): An instance of a concrete BaseModelStrategy.
        """
        if not isinstance(strategy, BaseModelStrategy):
            raise TypeError("Provided strategy must be an instance of BaseModelStrategy.")
        self._strategy = strategy
        self.trained_model = None

    def set_strategy(self, strategy: BaseModelStrategy):
        """
        Sets a new modeling strategy for the ModelTrainer.

        Parameters:
        strategy (BaseModelStrategy): The new strategy to be used for modeling.
        """
        if not isinstance(strategy, BaseModelStrategy):
            raise TypeError("Provided strategy must be an instance of BaseModelStrategy.")
        self._strategy = strategy
        self.trained_model = None # Reset trained model when strategy changes

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the model using the current strategy.

        Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        """
        print(f"--- Training Model: {self._strategy.name} ---")
        if X_train.empty or y_train.empty:
            print("Warning: Training data is empty. Skipping training.")
            return

        self._strategy.train(X_train, y_train)
        self.trained_model = self._strategy.get_model()
        print(f"Model '{self._strategy.name}' trained.")

    def predict_model(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained model of the current strategy.

        Parameters:
        X_test (pd.DataFrame): Features for prediction.

        Returns:
        np.ndarray: Array of predictions.
        """
        if self.trained_model is None:
            raise RuntimeError(f"Model '{self._strategy.name}' not trained. Call train_model() first.")
        if X_test.empty:
            print("Warning: Test data is empty. Returning empty predictions.")
            return np.array([])

        print(f"--- Generating Predictions with: {self._strategy.name} ---")
        predictions = self._strategy.predict(X_test)
        print(f"Predictions generated for {len(predictions)} samples.")
        return predictions

    def get_current_model_object(self) -> Any:
        """
        Returns the raw trained model object from the current strategy.
        """
        return self.trained_model

    def get_strategy_name(self) -> str:
        """
        Returns the name of the current modeling strategy.
        """
        return self._strategy.name
