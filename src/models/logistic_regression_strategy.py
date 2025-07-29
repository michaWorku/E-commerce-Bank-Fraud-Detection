import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from pathlib import Path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.base_model_strategy import BaseModelStrategy


class LogisticRegressionStrategy(BaseModelStrategy):
    """
    Concrete strategy for Logistic Regression model.
    """
    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initializes the Logistic Regression model.

        Parameters:
        random_state (int): Random seed for reproducibility.
        kwargs: Additional parameters for LogisticRegression.
        """
        super().__init__()
        # LogisticRegression is typically used for classification.
        # For binary classification, we often need to specify a solver and max_iter.
        # C is the inverse of regularization strength; smaller values specify stronger regularization.
        self.model = LogisticRegression(random_state=random_state, **kwargs)
        self._name = "Logistic Regression Classifier"

    @property
    def name(self) -> str:
        return self._name

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the Logistic Regression model.

        Parameters:
        X (pd.DataFrame): Training features.
        y (pd.Series): Target variable for training.
        """
        if X.empty or y.empty:
            print(f"Warning: Training data (X or y) is empty for {self.name}. Skipping training.")
            return

        print(f"Training {self.name} model...")
        self.model.fit(X, y)
        print(f"{self.name} training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained Logistic Regression model.
        For classification, returns class probabilities for the positive class.

        Parameters:
        X (pd.DataFrame): Features for prediction.

        Returns:
        np.ndarray: Array of predicted probabilities for the positive class.
        """
        if self.model is None:
            raise RuntimeError(f"{self.name} model not trained. Call train() first.")
        if X.empty:
            print(f"Warning: Prediction data (X) is empty for {self.name}. Returning empty array.")
            return np.array([])
            
        # For classification, we typically want probabilities for evaluation metrics like ROC-AUC
        # predict_proba returns probabilities for [class 0, class 1], we need class 1
        return self.model.predict_proba(X)[:, 1]

    def get_model(self):
        """
        Returns the trained Logistic Regression model object.
        """
        return self.model
