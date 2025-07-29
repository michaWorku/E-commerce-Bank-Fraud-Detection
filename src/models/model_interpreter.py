import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.class_names = class_names
        self.training_data_for_lime = training_data_for_lime # Store for LIME
        self.max_background_samples_shap = max_background_samples_shap # Store for SHAP sampling
        self.max_background_samples_lime = max_background_samples_lime # Store for LIME sampling

        if self.model_type not in ['regression', 'classification']:
            raise ValueError("model_type must be 'regression' or 'classification'.")
        if self.model_type == 'classification' and not self.class_names:
            raise ValueError("class_names must be provided for classification models.")
        
        # Initialize SHAP explainer
        self.shap_explainer = self._initialize_shap_explainer()

    def _initialize_shap_explainer(self):
        """Initializes the appropriate SHAP explainer based on model type."""
        
        # Determine if the model is a tree-based model (for efficiency with TreeExplainer)
        # Using string matching on model type is generally robust
        is_tree_model = any(
            x in str(type(self.model)) for x in ["XGB", "LGBM", "CatB", "DecisionTree", "RandomForest"]
        )

        if is_tree_model:
            print("Initializing TreeExplainer for SHAP (efficient for tree-based models).")
            # TreeExplainer is fast and can handle the full dataset, no explicit sampling needed here
            return shap.TreeExplainer(self.model)
        else:
            print("Initializing KernelExplainer for SHAP (model-agnostic, can be slower).")
            # For KernelExplainer, the background dataset needs to be sampled to prevent MemoryErrors.
            background_data = self.training_data_for_lime # Use the same training data as for LIME
            if background_data is None:
                print("Warning: training_data_for_lime not provided for KernelExplainer. "
                      "Using a small dummy background. SHAP values may be less reliable.")
                background_data = np.zeros((1, len(self.feature_names)))
            
            # If background data is a pandas DataFrame, convert to numpy array
            if isinstance(background_data, pd.DataFrame):
                background_data = background_data.values

            # Sample the background data if it's too large for KernelExplainer
            if background_data.shape[0] > self.max_background_samples_shap:
                print(f"Sampling KernelExplainer background data from {background_data.shape[0]} "
                      f"to {self.max_background_samples_shap} instances to reduce memory usage.")
                rng = np.random.default_rng(42) # Ensure reproducibility of sampling
                sample_indices = rng.choice(background_data.shape[0], self.max_background_samples_shap, replace=False)
                background_data_sampled = background_data[sample_indices, :]
            else:
                background_data_sampled = background_data
            
            # Ensure the background data is numerical (float) for KernelExplainer
            if not np.issubdtype(background_data_sampled.dtype, np.number):
                print(f"Warning: KernelExplainer background data is not purely numerical (dtype: {background_data_sampled.dtype}). Attempting to convert to float.")
                try:
                    background_data_sampled = background_data_sampled.astype(float)
                except ValueError:
                    print("Error: KernelExplainer background data cannot be converted to float. SHAP may fail.")
            
            # For classification, SHAP often works best with predict_proba
            if self.model_type == 'classification' and hasattr(self.model, 'predict_proba'):
                return shap.KernelExplainer(self.model.predict_proba, background_data_sampled)
            else:
                # For regression or classification models without predict_proba
                return shap.KernelExplainer(self.model.predict, background_data_sampled)


    def explain_model_shap(self, X_data: pd.DataFrame):
        """
        Generates SHAP values for the given data and stores them.

        Args:
            X_data (pd.DataFrame): The input data (e.g., test set) for SHAP explanation.
        """
        print(f"Generating SHAP explanations for {len(X_data)} instances...")
        
        # Convert to numpy array explicitly for compatibility with shap explainers
        if isinstance(X_data, pd.DataFrame):
            X_data_np = X_data.values
        else:
            X_data_np = X_data

        # Ensure the data for explanation is numerical (float)
        if not np.issubdtype(X_data_np.dtype, np.number):
            print(f"Warning: X_data for SHAP is not purely numerical (dtype: {X_data_np.dtype}). Attempting to convert to float.")
            try:
                X_data_np = X_data_np.astype(float)
            except ValueError:
                print("Error: X_data for SHAP cannot be converted to float. SHAP may fail.")

        if isinstance(self.shap_explainer, shap.TreeExplainer):
            # TreeExplainer can often handle DataFrame directly, but numpy array is safer universally.
            self.shap_values = self.shap_explainer.shap_values(X_data) # Use original X_data (DataFrame) for TreeExplainer if preferred, or X_data_np
        else:
            # For KernelExplainer, ensure data is numpy array
            self.shap_values = self.shap_explainer.shap_values(X_data_np)
        print("SHAP values generated.")

    def plot_shap_summary(self, X_data: pd.DataFrame):
        """
        Generates and displays a SHAP summary plot.
        Requires shap_values to be generated first.

        Args:
            X_data (pd.DataFrame): The input data (e.g., test set) used for SHAP explanation.
        """
        if not hasattr(self, 'shap_values'):
            print("Error: SHAP values not generated. Run explain_model_shap() first.")
            return

        print("Displaying SHAP summary plot...")
        plt.figure(figsize=(10, 6))
        
        # For classification, shap_values will be a list of arrays (one per class)
        if self.model_type == 'classification' and isinstance(self.shap_values, list):
            # For binary classification, typically explain class 1 (positive class)
            shap_values_to_plot = self.shap_values[1] 
            # If multi-class, can choose one, or sum absolute values
        else:
            shap_values_to_plot = self.shap_values

        shap.summary_plot(shap_values_to_plot, X_data, feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary Plot: Global Feature Importance')
        plt.tight_layout()
        plt.show()

    def explain_instance_lime(self, instance: pd.Series):
        """
        Generates and displays a LIME explanation for a single instance.

        Args:
            instance (pd.Series): A single data instance (row) to explain.
        """
        if self.training_data_for_lime is None:
            print("Error: training_data_for_lime was not provided during ModelInterpreter initialization. LIME cannot be used.")
            return

        print(f"Generating LIME explanation for instance (first few features):\n{instance.head().to_string()}") # Show head for brevity if long
        
        # Ensure training data for LIME is numerical NumPy array
        if isinstance(self.training_data_for_lime, pd.DataFrame):
            kernel_training_data = self.training_data_for_lime.values
        else:
            kernel_training_data = self.training_data_for_lime

        # Convert training data to float and handle potential NaNs before LIME initialization
        if not np.issubdtype(kernel_training_data.dtype, np.number):
             print(f"Warning: training_data_for_lime is not purely numerical (dtype: {kernel_training_data.dtype}). Attempting to convert to float for LIME.")
             try:
                 kernel_training_data = kernel_training_data.astype(float)
             except ValueError:
                 print("Error: training_data_for_lime cannot be converted to float. LIME requires numerical data. Skipping LIME explanation.")
                 return
        
        # LIME's internal NearestNeighbors can struggle with NaNs, so fill them or ensure data is clean
        if np.isnan(kernel_training_data).any():
            print("Warning: NaN values found in training_data_for_lime. LIME may behave unexpectedly. Filling with 0.")
            kernel_training_data = np.nan_to_num(kernel_training_data, nan=0.0) # Simple fill, more advanced strategies might be needed

        # --- NEW: Sample LIME background data if it's too large ---
        if kernel_training_data.shape[0] > self.max_background_samples_lime:
            print(f"Sampling LIME background data from {kernel_training_data.shape[0]} "
                  f"to {self.max_background_samples_lime} instances to reduce memory usage.")
            rng = np.random.default_rng(42) # Use same seed for reproducibility
            sample_indices = rng.choice(kernel_training_data.shape[0], self.max_background_samples_lime, replace=False)
            kernel_training_data_sampled = kernel_training_data[sample_indices, :]
        else:
            kernel_training_data_sampled = kernel_training_data


        # Ensure the instance to explain is a numpy array (LIME expects this)
        if isinstance(instance, pd.Series):
            instance_values = instance.values
        else:
            instance_values = instance
        
        # Convert instance to float and handle potential NaNs
        if not np.issubdtype(instance_values.dtype, np.number):
             print(f"Warning: Instance for LIME is not purely numerical (dtype: {instance_values.dtype}). Attempting to convert to float.")
             try:
                 instance_values = instance_values.astype(float)
             except ValueError:
                 print("Error: Instance for LIME cannot be converted to float. LIME may fail.")
                 return
        if np.isnan(instance_values).any():
            print("Warning: NaN values found in instance. LIME may behave unexpectedly. Filling with 0.")
            instance_values = np.nan_to_num(instance_values, nan=0.0)

        # Initialize LimeTabularExplainer internally
        # LIME takes the *raw* training data for its background, now sampled
        # It's important that this training_data is numerically processed.
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=kernel_training_data_sampled, # Use sampled data here
            feature_names=self.feature_names,
            class_names=self.class_names if self.model_type == 'classification' else ['Prediction'],
            mode=self.model_type
        )

        # Generate explanation
        predict_fn = self.model.predict_proba if self.model_type == 'classification' and hasattr(self.model, 'predict_proba') else self.model.predict

        explanation = explainer.explain_instance(
            data_row=instance_values,
            predict_fn=predict_fn,
            num_features=min(10, len(self.feature_names)) # Limit features for readability
        )
        
        # Print explanation as text
        print("\nLIME Explanation (Feature Contribution to Prediction):")
        for feature, weight in explanation.as_list():
            print(f"  {feature}: {weight:.4f}")

        # Optional: Display explanation plot (if running in notebook environment)
        explanation.as_pyplot_figure()
        plt.title('LIME Explanation for Single Instance')
        plt.tight_layout()
        plt.show()
