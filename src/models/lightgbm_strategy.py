import pandas as pd
import numpy as np
import lightgbm as lgb

from pathlib import Path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.base_model_strategy import BaseModelStrategy


class LightGBMStrategy(BaseModelStrategy):
    """
    Concrete strategy for LightGBM model (Regressor or Classifier).
    """
    def __init__(self, model_type: str = 'regressor', random_state: int = 42, **kwargs):
        """
        Initializes the LightGBM model.

        Parameters:
        model_type (str): Type of model to use: 'regressor' or 'classifier'.
        random_state (int): Random seed for reproducibility.
        kwargs: Additional parameters for lgb.LGBMRegressor or lgb.LGBMClassifier.
        """
        super().__init__()
        self.model_type = model_type
        self._name = "LightGBM Regressor" if model_type == 'regressor' else "LightGBM Classifier"

        if model_type == 'regressor':
            self.model = lgb.LGBMRegressor(random_state=random_state, **kwargs)
        elif model_type == 'classifier':
            self.model = lgb.LGBMClassifier(random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'regressor' or 'classifier'.")

    @property
    def name(self) -> str:
        return self._name

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the LightGBM model.

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
        Makes predictions using the trained LightGBM model.

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
        Returns the trained LightGBM model object.
        """
        return self.model
