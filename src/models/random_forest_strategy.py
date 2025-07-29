import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from pathlib import Path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.base_model_strategy import BaseModelStrategy


class RandomForestStrategy(BaseModelStrategy):
    """
    Concrete strategy for Random Forest model (Regressor or Classifier).
    """
    def __init__(self, model_type: str = 'regressor', n_estimators: int = 100, random_state: int = 42, **kwargs):
        """
        Initializes the Random Forest model.

        Parameters:
        model_type (str): Type of model to use: 'regressor' or 'classifier'.
        n_estimators (int): The number of trees in the forest.
        random_state (int): Random seed for reproducibility.
        kwargs: Additional parameters for RandomForestRegressor or RandomForestClassifier.
        """
        super().__init__()
        self.model_type = model_type
        self._name = "Random Forest Regressor" if model_type == 'regressor' else "Random Forest Classifier"

        if model_type == 'regressor':
            self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, **kwargs)
        elif model_type == 'classifier':
            self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'regressor' or 'classifier'.")

    @property
    def name(self) -> str:
        return self._name

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the Random Forest model.

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
        Makes predictions using the trained Random Forest model.

        Parameters:
        X (pd.DataFrame): Features for prediction.

        Returns:
        np.ndarray: Array of predictions. For classification, returns class probabilities for positive class.
        """
        if self.model is None:
            raise RuntimeError(f"{self.name} model not trained. Call train() first.")
        if X.empty:
            print(f"Warning: Prediction data (X) is empty for {self.name}. Returning empty array.")
            return np.array([])
            
        if self.model_type == 'classifier':
            # For classification, return probabilities for the positive class (class 1)
            # This is crucial for metrics like ROC-AUC
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def get_model(self):
        """
        Returns the trained Random Forest model object.
        """
        return self.model

