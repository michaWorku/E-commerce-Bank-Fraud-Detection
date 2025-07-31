import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

class ModelInterpreter:
    """
    A class for interpreting machine learning models using SHAP and LIME.
    Supports both regression and classification models.
    """
    def __init__(self, model, feature_names: list, model_type: str, class_names: list = None, training_data_for_lime=None, max_background_samples_shap: int = 500, max_background_samples_lime: int = 1000):
        """
        Initializes the ModelInterpreter.

        Args:
            model: The trained machine learning model.
            feature_names (list): A list of feature names corresponding to the input data.
            model_type (str): Type of model ('regression' or 'classification').
            class_names (list, optional): List of class names for classification models.
                                          Required if model_type is 'classification'.
            training_data_for_lime: The training data (as a NumPy array or DataFrame) used
                                    to train the model, required for LIME's background data.
            max_background_samples_shap (int): Maximum number of samples to use for SHAP's KernelExplainer
                                               background data, to prevent MemoryErrors.
            max_background_samples_lime (int): Maximum number of samples to use for LIME's
                                               background data, to prevent MemoryErrors.
        """
        if model_type not in ['regression', 'classification']:
            raise ValueError("model_type must be 'regression' or 'classification'.")
        if model_type == 'classification' and class_names is None:
            raise ValueError("class_names must be provided for classification models.")

        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.class_names = class_names
        self.training_data_for_lime = training_data_for_lime
        self.max_background_samples_shap = max_background_samples_shap
        self.max_background_samples_lime = max_background_samples_lime

        # Initialize SHAP explainer
        is_tree_model = isinstance(self.model, (
            DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier,
            xgb.XGBClassifier, xgb.XGBRegressor
        ))

        if is_tree_model:
            print("Initializing TreeExplainer for SHAP (efficient for tree-based models).")
            self.explainer_shap = shap.TreeExplainer(self.model)
        else:
            print("Initializing KernelExplainer for SHAP (general purpose).")
            if self.training_data_for_lime is not None and len(self.training_data_for_lime) > self.max_background_samples_shap:
                print(f"Sampling {self.max_background_samples_shap} background samples for SHAP KernelExplainer.")
                if isinstance(self.training_data_for_lime, np.ndarray):
                    background_data = pd.DataFrame(self.training_data_for_lime, columns=self.feature_names).sample(
                        n=self.max_background_samples_shap, random_state=42
                    )
                else:
                    background_data = self.training_data_for_lime.sample(
                        n=self.max_background_samples_shap, random_state=42
                    )
            elif self.training_data_for_lime is not None:
                background_data = self.training_data_for_lime
            else:
                raise ValueError("training_data_for_lime must be provided for KernelExplainer if not a tree model.")
            
            if isinstance(background_data, pd.DataFrame):
                background_data = background_data.values

            # Ensure predict_proba is used for classification if available, otherwise predict
            if self.model_type == 'classification' and hasattr(self.model, 'predict_proba'):
                predict_fn_for_shap = self.model.predict_proba
            else:
                predict_fn_for_shap = self.model.predict

            self.explainer_shap = shap.KernelExplainer(predict_fn_for_shap, background_data)

    def explain_model_shap(self, X_data: pd.DataFrame):
        """
        Generates SHAP values for the entire dataset or a sample of it.

        Parameters:
        X_data (pd.DataFrame): The dataset (or a sample) for which to generate SHAP values.
        """
        if X_data.empty:
            print("Input DataFrame for SHAP is empty. Skipping SHAP explanation.")
            return

        print(f"Generating SHAP explanations for {len(X_data)} instances...")
        shap_values = self.explainer_shap.shap_values(X_data)
        
        if self.model_type == 'classification':
            self.shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            self.shap_values = shap_values
        print("SHAP values generated.")

    def plot_shap_summary(self, X_data: pd.DataFrame):
        """
        Plots a SHAP summary plot (e.g., beeswarm or bar).
        """
        if self.shap_values is None:
            print("SHAP values not generated. Call explain_model_shap() first.")
            return
        
        print("Displaying SHAP summary plot...")
        shap.summary_plot(self.shap_values, X_data, feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        plt.show()

    def explain_instance_lime(self, instance: np.ndarray):
        """
        Generates and prints a LIME explanation for a single instance.

        Parameters:
        instance (np.ndarray): A single data instance (row) as a NumPy array.
        """
        if instance.size == 0:
            print("Input instance for LIME is empty. Skipping LIME explanation.")
            return

        if self.training_data_for_lime is None or self.training_data_for_lime.size == 0:
            print("LIME training data (background data) is not available or empty. Cannot perform LIME explanation.")
            return

        instance_values = np.array(instance).flatten()

        if len(instance_values) != len(self.feature_names):
            print(f"ERROR: Instance to explain has {len(instance_values)} features, but expected {len(self.feature_names)}. Cannot perform LIME explanation.")
            return

        kernel_training_data_sampled = self.training_data_for_lime
        if kernel_training_data_sampled.ndim == 1:
            kernel_training_data_sampled = kernel_training_data_sampled.reshape(1, -1)

        if np.isnan(instance_values).any():
            print("Warning: NaNs found in instance to explain for LIME. This can cause LIME to behave unexpectedly. Filling with 0.")
            instance_values = np.nan_to_num(instance_values, nan=0.0)

        # Determine predict_fn for LIME based on model_type and availability
        if self.model_type == 'classification':
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                # This is the critical check that caused the error.
                # If predict_proba is missing for a classifier, LIME cannot proceed.
                raise NotImplementedError(f"Classifier model '{type(self.model).__name__}' does not have 'predict_proba' method, which is required for LIME classification explanation. "
                                          "Please ensure you are using a classifier that provides probability scores.")
        else: # Regression
            predict_fn = self.model.predict

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=kernel_training_data_sampled,
            feature_names=self.feature_names,
            class_names=self.class_names if self.model_type == 'classification' else ['Prediction'],
            mode=self.model_type
        )

        explanation = explainer.explain_instance(
            data_row=instance_values,
            predict_fn=predict_fn,
            num_features=min(10, len(self.feature_names))
        )
        
        print("LIME Explanation (Feature Contribution to Prediction):")
        print(f"Instance values (first 5 features): {instance_values[:5]}")
        for feature, weight in explanation.as_list():
            print(f"  {feature}: {weight:.4f}")

        explanation.as_pyplot_figure()
        plt.title('LIME Explanation for Single Instance')
        plt.tight_layout()
        plt.show()
