import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import sys
from tqdm import tqdm 
import cloudpickle
import os

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_processing.loader import load_data
from src.data_processing.preprocessor import FraudDataProcessor, CreditCardDataProcessor # Import specific processors
from src.utils.helpers import merge_ip_to_country # Needed for FraudData preprocessing

# Define paths
RAW_DATA_DIR = project_root / "data" / "raw"
MLRUNS_PATH = project_root / "mlruns" # MLflow tracking directory
PREDICTIONS_DIR = project_root / "data" / "predictions"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure MLflow tracking URI is set
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")

def predict_risk(
    data_path: Path, 
    dataset_type: str, # 'ecommerce' or 'creditcard'
    registered_model_name: str = None, # Will be derived from dataset_type if None
    model_version: str = "latest"
):
    """
    Loads a registered model and its associated data processor from MLflow,
    and uses them to predict risk on new data.

    Args:
        data_path (Path): Path to the new data CSV file.
        dataset_type (str): Specifies which dataset's model/processor to use ('ecommerce' or 'creditcard').
        registered_model_name (str): The name of the registered model in MLflow. If None, derived from dataset_type.
        model_version (str): The version of the model to load (e.g., '1', 'latest').
    """
    if dataset_type not in ['ecommerce', 'creditcard']:
        print("Error: dataset_type must be 'ecommerce' or 'creditcard'.")
        return

    if registered_model_name is None:
        if dataset_type == 'ecommerce':
            # Assuming XGBoost was chosen as best for E-commerce in run_train.py
            registered_model_name = "FraudDetection_XGBoost" 
        else: # dataset_type == 'creditcard'
            # Assuming XGBoost was chosen as best for Credit Card in run_train.py
            registered_model_name = "CreditCardFraud_XGBoost"

    print(f"Starting prediction process for {dataset_type} data using model '{registered_model_name}' version '{model_version}'...")

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
        loaded_model = mlflow.sklearn.load_model(model_uri) # Use mlflow.sklearn.load_model for sklearn models
        print("Model loaded successfully.")

        # Load the associated DataProcessor artifact
        processor_uri = f"runs:/{model_run_id}/processor"
        processor_local_path = mlflow.artifacts.download_artifacts(processor_uri)
        with open(processor_local_path, "rb") as f:
            loaded_processor = cloudpickle.load(f)
        print("DataProcessor loaded successfully.")

    except Exception as e:
        print(f"Error loading model or processor from MLflow Registry: {e}")
        print("Please ensure the model and its processor are registered and the MLflow tracking server is accessible.")
        return

    # --- 2. Load and Preprocess New Data ---
    print(f"Loading new raw data from {data_path}...")
    new_df_raw = load_data(data_path, delimiter=',')
    if new_df_raw is None or new_df_raw.empty:
        print("Error: New data could not be loaded or is empty. Exiting prediction.")
        return

    # Apply initial transformations specific to the dataset type if necessary,
    # before passing to the loaded_processor's transform method.
    X_new_raw_for_processing = new_df_raw.copy()
    if dataset_type == 'ecommerce':
        # Apply renaming and IP merge as done during training for FraudDataProcessor
        X_new_raw_for_processing = X_new_raw_for_processing.rename(columns={
            'user_id': 'CustomerId',
            'purchase_value': 'Amount',
            'purchase_time': 'TransactionStartTime',
            'user_id': 'TransactionId'
        })
        # Need to load IP_TO_COUNTRY_PATH if it's not part of the processor's state
        # For simplicity in this script, assuming IP merge is handled by external call
        # or that the processor itself encapsulates it if needed.
        # If the processor expects a specific column like 'ip_address' to be present,
        # ensure it's merged here.
        # For this example, we'll assume new_transactions.csv might not have signup_time, etc.
        # We need to ensure the structure matches what the processor expects.
        # This is simplified. In a real scenario, the processor might need to be more robust
        # to handle missing columns in inference data or you'd need to ensure the input data
        # matches the training data format.
        
        # For the dummy data, 'ip_address' and 'signup_time' are not directly in new_transactions.csv
        # If the processor expects them, we need to add dummy versions or ensure they are processed.
        # The FraudDataProcessor expects 'ip_address' and 'signup_time'.
        # Let's add dummy columns if they are missing for the new data.
        if 'signup_time' not in X_new_raw_for_processing.columns:
            X_new_raw_for_processing['signup_time'] = pd.NaT # Or a default datetime
        if 'ip_address' not in X_new_raw_for_processing.columns:
            X_new_raw_for_processing['ip_address'] = np.nan # Or a default IP
        
        # Merge IP to country if the processor expects it
        # This requires ip_to_country_path to be available.
        ip_to_country_path = RAW_DATA_DIR / "IpAddress_to_Country.csv"
        ip_country_df_raw = load_data(ip_to_country_path, use_gpu=False)
        if not ip_country_df_raw.empty:
            X_new_raw_for_processing = merge_ip_to_country(X_new_raw_for_processing, ip_country_df_raw.copy())
        else:
            print("Warning: IP_TO_COUNTRY_PATH not found for E-commerce data. Skipping IP merge.")
            # Ensure 'country' column exists, perhaps as 'Unknown'
            if 'country' not in X_new_raw_for_processing.columns:
                X_new_raw_for_processing['country'] = 'Unknown'

    print("Processing new data using the loaded preprocessor...")
    X_predict_processed = loaded_processor.transform(X_new_raw_for_processing)

    # Ensure feature columns match those used during training
    # The loaded_processor should output consistent columns.
    # If there are still non-numeric columns, handle them.
    non_numeric_cols = X_predict_processed.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found in processed new data: {non_numeric_cols}. Attempting to convert/drop.")
        for col in non_numeric_cols:
            try:
                X_predict_processed[col] = pd.to_numeric(X_predict_processed[col], errors='coerce').fillna(0.0)
            except Exception:
                X_predict_processed = X_predict_processed.drop(columns=[col])

    if X_predict_processed.isnull().any().any():
        print("Warning: NaNs detected in features for prediction. Imputing with median/0.")
        for col in X_predict_processed.columns:
            if X_predict_processed[col].isnull().any():
                if X_predict_processed[col].dtype == 'object':
                    X_predict_processed[col] = X_predict_processed[col].fillna(X_predict_processed[col].mode()[0] if not X_predict_processed[col].mode().empty else 'missing')
                elif pd.api.types.is_numeric_dtype(X_predict_processed[col]):
                    X_predict_processed[col] = X_predict_processed[col].fillna(X_predict_processed[col].median() if not X_predict_processed[col].empty else 0.0)
                else:
                    X_predict_processed[col] = X_predict_processed[col].fillna(0)


    # --- 3. Make Predictions ---
    print("Making predictions on new data...")
    predictions_proba = loaded_model.predict(X_predict_processed)

    risk_predictions = (predictions_proba >= 0.5).astype(int)

    new_df_raw['predicted_risk_proba'] = predictions_proba
    new_df_raw['predicted_risk_label'] = risk_predictions

    print("\nPredictions complete. Sample of results:")
    # Display relevant columns, ensure they exist
    display_cols = ['TransactionId', 'CustomerId', 'Amount', 'predicted_risk_proba', 'predicted_risk_label']
    existing_display_cols = [col for col in display_cols if col in new_df_raw.columns]
    print(new_df_raw[existing_display_cols].head())

    predictions_output_path = PREDICTIONS_DIR / f"new_data_predictions_{dataset_type}.csv"
    new_df_raw.to_csv(predictions_output_path, index=False)
    print(f"\nPredictions saved to: {predictions_output_path}")

if __name__ == "__main__":
    # Example usage for E-commerce Fraud data
    ecommerce_dummy_data_path = RAW_DATA_DIR / "new_ecommerce_transactions.csv"
    if not ecommerce_dummy_data_path.exists():
        print(f"Creating a dummy new E-commerce data file at {ecommerce_dummy_data_path} for demonstration.")
        dummy_data_ecommerce = {
            'user_id': ['U101', 'U102', 'U103', 'U104'],
            'signup_time': ['2023-01-01 10:00:00', '2023-01-05 12:00:00', '2023-01-10 09:00:00', '2023-01-15 11:00:00'],
            'purchase_time': ['2024-01-01 08:00:00', '2024-01-05 13:00:00', '2024-01-10 10:00:00', '2024-01-12 11:00:00'],
            'purchase_value': [60.0, 110.0, 400.0, 90.0],
            'device_id': ['D1', 'D2', 'D1', 'D3'],
            'source': ['SEO', 'Ads', 'Direct', 'SEO'],
            'browser': ['Chrome', 'Firefox', 'Safari', 'Chrome'],
            'sex': ['M', 'F', 'M', 'F'],
            'age': [25, 30, 45, 28],
            'ip_address': [1000000000, 1000000001, 1000000002, 1000000003]
        }
        pd.DataFrame(dummy_data_ecommerce).to_csv(ecommerce_dummy_data_path, index=False)
        print("Dummy E-commerce data created.")
    
    # Run prediction for E-commerce data
    predict_risk(ecommerce_dummy_data_path, dataset_type='ecommerce')

    print("\n" + "="*50 + "\n")

    # Example usage for Credit Card Fraud data
    creditcard_dummy_data_path = RAW_DATA_DIR / "new_creditcard_transactions.csv"
    if not creditcard_dummy_data_path.exists():
        print(f"Creating a dummy new Credit Card data file at {creditcard_dummy_data_path} for demonstration.")
        # Create dummy data matching creditcard.csv structure (V1-V28, Time, Amount)
        dummy_data_creditcard = {f'V{i}': np.random.rand(4) for i in range(1, 29)}
        dummy_data_creditcard['Time'] = [100, 200, 300, 400] # Dummy time
        dummy_data_creditcard['Amount'] = [50.0, 120.0, 30.0, 250.0] # Dummy amount
        # Add a dummy TransactionId and CustomerId for consistency with output, though not in original creditcard.csv
        dummy_data_creditcard['TransactionId'] = ['CC_T1', 'CC_T2', 'CC_T3', 'CC_T4']
        dummy_data_creditcard['CustomerId'] = ['CC_C1', 'CC_C2', 'CC_C3', 'CC_C4']
        pd.DataFrame(dummy_data_creditcard).to_csv(creditcard_dummy_data_path, index=False)
        print("Dummy Credit Card data created.")

    # Run prediction for Credit Card data
    predict_risk(creditcard_dummy_data_path, dataset_type='creditcard')
