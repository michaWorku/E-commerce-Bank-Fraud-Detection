import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cloudpickle
import os

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_processing.loader import load_data
from src.data_processing.preprocessor import FraudDataProcessor, CreditCardDataProcessor # Import specific processors
from src.models.model_interpreter import ModelInterpreter
from src.utils.helpers import merge_ip_to_country # Needed for FraudData preprocessing

# Define paths
RAW_DATA_DIR = project_root / "data" / "raw"
MLRUNS_PATH = project_root / "mlruns" # MLflow tracking directory

# Ensure MLflow tracking URI is set
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")

def run_model_interpretation(
    dataset_type: str, # 'ecommerce' or 'creditcard'
    registered_model_name: str = None, # Will be derived from dataset_type if None
    model_version: str = "latest",
    num_samples_for_shap: int = 500, # Number of samples for global SHAP explanation
    num_instances_for_lime: int = 5 # Number of individual instances for LIME explanation
):
    """
    Loads a registered model and its associated data processor from MLflow,
    and performs model interpretation using SHAP and LIME.

    Args:
        dataset_type (str): Specifies which dataset's model/processor to use ('ecommerce' or 'creditcard').
        registered_model_name (str): The name of the registered model in MLflow. If None, derived from dataset_type.
        model_version (str): The version of the model to load (e.g., '1', 'latest').
        num_samples_for_shap (int): Number of samples from the test set to use for SHAP summary plot.
        num_instances_for_lime (int): Number of individual instances from the test set to explain with LIME.
    """
    if dataset_type not in ['ecommerce', 'creditcard']:
        print("Error: dataset_type must be 'ecommerce' or 'creditcard'.")
        return

    if registered_model_name is None:
        if dataset_type == 'ecommerce':
            registered_model_name = "FraudDetection_XGBoost" # Assuming XGBoost was chosen as best
        else: # dataset_type == 'creditcard'
            registered_model_name = "CreditCardFraud_XGBoost" # Assuming XGBoost was chosen as best

    print(f"Starting model interpretation process for {dataset_type} data...")

    # --- 1. Load Registered Model and its associated DataProcessor ---
    print(f"Loading model '{registered_model_name}' version '{model_version}' from MLflow Registry...")
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get the model version object to find the run_id
        if model_version == "latest":
            latest_versions = client.get_latest_versions(registered_model_name, stages=["None", "Staging", "Production"])
            if not latest_versions:
                raise ValueError(f"No versions found for model '{registered_model_name}' in any stage.")
            latest_version_obj = max(latest_versions, key=lambda mv: int(mv.version))
            actual_model_version = latest_version_obj.version
            print(f"Resolved 'latest' to actual version: {actual_model_version}")
        else:
            actual_model_version = model_version
            
        model_version_obj = client.get_model_version(registered_model_name, actual_model_version)
        model_run_id = model_version_obj.run_id

        # Load the model itself
        model_uri = f"runs:/{model_run_id}/model"
        sklearn_loaded_model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")

        # Load the associated DataProcessor artifact
        processor_uri = f"runs:/{model_run_id}/processor"
        processor_local_path = mlflow.artifacts.download_artifacts(processor_uri)
        with open(processor_local_path, "rb") as f:
            loaded_processor = cloudpickle.load(f)
        print("DataProcessor loaded successfully.")

    except Exception as e:
        print(f"Error loading model or processor from MLflow Registry for interpretation: {e}")
        print("Please ensure the model and its processor are registered and the MLflow tracking server is accessible.")
        return

    # --- 2. Load Raw Data and Preprocess for Interpretation ---
    print(f"Loading raw data for {dataset_type} for processing setup...")
    if dataset_type == 'ecommerce':
        raw_data_path = RAW_DATA_DIR / "Fraud_Data.csv"
        ip_to_country_path = RAW_DATA_DIR / "IpAddress_to_Country.csv"
        df_raw = load_data(raw_data_path, delimiter=',')
        ip_country_df_raw = load_data(ip_to_country_path, use_gpu=False)
        
        if df_raw.empty or ip_country_df_raw.empty:
            print(f"Error: Raw data for {dataset_type} could not be loaded or is empty. Exiting interpretation.")
            return

        # Apply initial renaming and IP merge as done during training
        df_merged = df_raw.rename(columns={
            'user_id': 'CustomerId',
            'purchase_value': 'Amount',
            'purchase_time': 'TransactionStartTime',
            'user_id': 'TransactionId'
        }).copy()
        df_merged = merge_ip_to_country(df_merged, ip_country_df_raw.copy())
        target_col = 'class'
        df_merged[target_col] = df_raw[target_col] # Re-attach target

        X_raw_for_processing = df_merged.drop(columns=[target_col]).copy()
        y_raw_for_processing = df_merged[target_col].copy()

    else: # dataset_type == 'creditcard'
        raw_data_path = RAW_DATA_DIR / "creditcard.csv"
        df_raw = load_data(raw_data_path, delimiter=',')
        if df_raw.empty:
            print(f"Error: Raw data for {dataset_type} could not be loaded or is empty. Exiting interpretation.")
            return
        target_col = 'Class'
        X_raw_for_processing = df_raw.drop(columns=[target_col]).copy()
        y_raw_for_processing = df_raw[target_col].copy()

    print(f"Processing data for interpretation using the loaded preprocessor for {dataset_type}...")
    X_processed_full = loaded_processor.transform(X_raw_for_processing)

    # Ensure feature columns match those used during training
    # Remove any non-numeric columns that might have slipped through
    non_numeric_cols = X_processed_full.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found in processed data for interpretation: {non_numeric_cols}. Dropping them.")
        X_processed_full = X_processed_full.drop(columns=non_numeric_cols)

    if X_processed_full.isnull().any().any():
        print("Warning: NaNs detected in final feature set for interpretation. Imputing with median/0.")
        for col in X_processed_full.columns:
            if X_processed_full[col].isnull().any():
                if X_processed_full[col].dtype == 'object':
                    X_processed_full[col] = X_processed_full[col].fillna(X_processed_full[col].mode()[0] if not X_processed_full[col].mode().empty else 'missing')
                elif pd.api.types.is_numeric_dtype(X_processed_full[col]):
                    X_processed_full[col] = X_processed_full[col].fillna(X_processed_full[col].median() if not X_processed_full[col].empty else 0.0)
                else:
                    X_processed_full[col] = X_processed_full[col].fillna(0)

    # Split to get a test set for interpretation, ensuring stratification
    _, X_test, _, y_test = train_test_split(
        X_processed_full, y_raw_for_processing, test_size=0.2, random_state=42, stratify=y_raw_for_processing
    )
    print(f"Using {len(X_test)} samples from test set for interpretation.")

    # --- 3. Initialize Model Interpreter ---
    # Sample training data for LIME background if it's too large
    # For interpretation, it's often better to use a representative sample of the *processed* training data.
    # We can use X_processed_full for this, as it represents the full processed feature space.
    training_data_for_lime_bg = X_processed_full.sample(n=min(loaded_processor.max_background_samples_lime, len(X_processed_full)), random_state=42).values

    interpreter = ModelInterpreter(
        model=sklearn_loaded_model,
        feature_names=X_test.columns.tolist(),
        model_type='classification',
        class_names=['Non-Fraud', 'Fraud'],
        training_data_for_lime=training_data_for_lime_bg,
        max_background_samples_shap=num_samples_for_shap,
        max_background_samples_lime=loaded_processor.max_background_samples_lime # Use processor's configured limit
    )

    # --- 4. Global Interpretation (SHAP) ---
    print("\n--- Performing Global Interpretation with SHAP ---")
    X_shap_sample = X_test.sample(n=min(num_samples_for_shap, len(X_test)), random_state=42)
    
    interpreter.explain_model_shap(X_shap_sample)
    interpreter.plot_shap_summary(X_shap_sample)

    # --- 5. Local Interpretation (LIME) ---
    print(f"\n--- Performing Local Interpretation with LIME for {num_instances_for_lime} instances ---")
    # Find actual fraudulent and non-fraudulent instances in the test set for LIME
    fraud_indices = y_test[y_test == 1].index
    non_fraud_indices = y_test[y_test == 0].index

    # Explain a sample fraudulent transaction
    if not fraud_indices.empty:
        for i in tqdm(range(min(num_instances_for_lime // 2, len(fraud_indices))), desc="LIME Fraud Explanations"):
            instance_to_explain = X_test.loc[fraud_indices[i]]
            interpreter.explain_instance_lime(instance_to_explain)
            plt.show() # Display plot for each instance
    else:
        print("No fraudulent instances found in the test set for LIME explanation.")

    # Explain a sample non-fraudulent transaction
    if not non_fraud_indices.empty:
        for i in tqdm(range(min(num_instances_for_lime - (num_instances_for_lime // 2), len(non_fraud_indices))), desc="LIME Non-Fraud Explanations"):
            instance_to_explain = X_test.loc[non_fraud_indices[i]]
            interpreter.explain_instance_lime(instance_to_explain)
            plt.show() # Display plot for each instance
    else:
        print("No non-fraudulent instances found in the test set for LIME explanation.")

    print("\nModel interpretation process completed.")


if __name__ == "__main__":
    # Example for E-commerce Fraud data
    print("Running interpretation for E-commerce Fraud data...")
    run_model_interpretation(dataset_type='ecommerce')

    print("\n" + "="*50 + "\n")

    # Example for Credit Card Fraud data
    print("Running interpretation for Credit Card Fraud data...")
    run_model_interpretation(dataset_type='creditcard')
