import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from pathlib import Path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.base_model_strategy import BaseModelStrategy

class DecisionTreeStrategy(BaseModelStrategy):
    """
    Concrete strategy for Decision Tree model (Regressor or Classifier).
    """
    def __init__(self, model_type: str = 'regressor', random_state: int = 42, **kwargs):
        """
        Initializes the Decision Tree model.

        Parameters:
        model_type (str): Type of model to use: 'regressor' or 'classifier'.
        random_state (int): Random seed for reproducibility.
        kwargs: Additional parameters for DecisionTreeRegressor or DecisionTreeClassifier.
        """
        super().__init__()
        self.model_type = model_type
        self._name = "Decision Tree Regressor" if model_type == 'regressor' else "Decision Tree Classifier"

        if model_type == 'regressor':
            self.model = DecisionTreeRegressor(random_state=random_state, **kwargs)
        elif model_type == 'classifier':
            self.model = DecisionTreeClassifier(random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'regressor' or 'classifier'.")

    @property
    def name(self) -> str:
        return self._name

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the Decision Tree model.

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
        Makes predictions using the trained Decision Tree model.

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
            return self.model.predict_proba(X)[:, 1] # Return probabilities for the positive class
        return self.model.predict(X)

    def get_model(self):
        """
        Returns the trained Decision Tree model object.
        """
        return self.model
