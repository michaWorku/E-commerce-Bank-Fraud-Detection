import pandas as pd
import numpy as np
import mlflow # Keep mlflow import for potential future use or context, but not for direct loading
from pathlib import Path
import sys
from tqdm import tqdm
import cloudpickle
import os
import argparse
import shutil # For cleaning up temporary directories if any are still created by other parts of mlflow

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Fix import path for load_data
from src.data_processing.loader import load_data, get_relative_path
from src.data_processing.preprocessor import FraudDataProcessor, CreditCardDataProcessor # Import specific processors
from src.utils.helpers import merge_ip_to_country # Needed for FraudData preprocessing

# Define paths
RAW_DATA_DIR = project_root / "data" / "raw"
MLRUNS_PATH = project_root / "mlruns" # MLflow tracking directory (still needed for set_tracking_uri if using MLflow at all)
PREDICTIONS_DIR = project_root / "data" / "predictions"
EXPORTED_MODELS_DIR = project_root / "exported_models" # Directory where models are directly exported

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTED_MODELS_DIR.mkdir(parents=True, exist_ok=True) # Ensure this exists

# Ensure MLflow tracking URI is set (even if not loading via MLflow.artifacts, it's good practice)
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")

# --- Column Definitions for Fraud_Data.csv (must match run_train.py) ---
FRAUD_TARGET_COL = 'class'
FRAUD_NUMERICAL_FEATURES = ['purchase_value', 'age']
FRAUD_CATEGORICAL_FEATURES = ['source', 'browser', 'sex', 'country'] # 'country' is added after merge
FRAUD_PURCHASE_TIME_COL = 'purchase_time'
FRAUD_SIGNUP_TIME_COL = 'signup_time'
FRAUD_AMOUNT_COL = 'purchase_value' # The column to treat as 'Amount' for HandleAmountFeatures
FRAUD_ID_COLS_FOR_AGG = ['user_id', 'device_id', 'ip_address'] # IDs for frequency/velocity

# --- Column Definitions for creditcard.csv (must match run_train.py) ---
CREDITCARD_TARGET_COL = 'Class'
CREDITCARD_NUMERICAL_FEATURES = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']


def predict_risk(
    data_path: Path,
    dataset_type: str, # 'ecommerce' or 'creditcard'
    # registered_model_name and model_version are no longer strictly used for loading,
    # but can be kept as arguments for consistency or if MLflow loading is re-enabled.
    # We will derive the file paths directly from dataset_type.
    registered_model_name: str = None,
    model_version: str = "latest"
):
    """
    Loads the best-performing model and its associated data processor directly from
    the 'exported_models' directory and uses them to predict risk on new data.

    Args:
        data_path (Path): Path to the new data CSV file.
        dataset_type (str): Specifies which dataset's model/processor to use ('ecommerce' or 'creditcard').
        registered_model_name (str): (Ignored for direct loading)
        model_version (str): (Ignored for direct loading)
    """
    if dataset_type not in ['ecommerce', 'creditcard']:
        print("Error: dataset_type must be 'ecommerce' or 'creditcard'.")
        return

    print(f"Starting prediction process for {dataset_type} data by loading directly from exported_models...")

    # Determine paths for the exported model and processor
    if dataset_type == 'ecommerce':
        model_path = EXPORTED_MODELS_DIR / "best_ecommerce_model.pkl"
        processor_path = EXPORTED_MODELS_DIR / "best_ecommerce_processor.pkl"
    else: # dataset_type == 'creditcard'
        model_path = EXPORTED_MODELS_DIR / "best_creditcard_model.pkl"
        processor_path = EXPORTED_MODELS_DIR / "best_creditcard_processor.pkl"

    # --- 1. Load Model and its associated DataProcessor directly from .pkl files ---
    loaded_model = None
    loaded_processor = None
    try:
        print(f"Loading model from: {get_relative_path(model_path)}")
        with open(model_path, "rb") as f:
            loaded_model = cloudpickle.load(f)
        print("Model loaded successfully.")

        print(f"Loading DataProcessor from: {get_relative_path(processor_path)}")
        with open(processor_path, "rb") as f:
            loaded_processor = cloudpickle.load(f)
        print("DataProcessor loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Required model or processor file not found. Please ensure `run_train.py` has been executed to export the models to {get_relative_path(EXPORTED_MODELS_DIR)}.")
        print(f"Missing file: {get_relative_path(model_path)} or {get_relative_path(processor_path)}")
        return
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        return

    if loaded_model is None or loaded_processor is None:
        print("Failed to load model or processor. Exiting prediction.")
        return

    # --- 2. Load and Preprocess New Data ---
    print(f"Loading new raw data from {get_relative_path(data_path)}...")
    new_df_raw = load_data(data_path, delimiter=',')
    if new_df_raw.empty:
        print("Error: New data could not be loaded or is empty. Exiting prediction.")
        return

    # --- IMPORTANT: Create TransactionId and CustomerId on new_df_raw FIRST ---
    # This ensures they are correctly indexed and available for the final output.
    if dataset_type == 'ecommerce':
        # For e-commerce, user_id is the primary identifier. Let's use it for CustomerId.
        # And create a TransactionId based on user_id and original index.
        if 'user_id' in new_df_raw.columns:
            new_df_raw['CustomerId'] = new_df_raw['user_id'].astype(str)
            # Robustly create TransactionId by concatenating CustomerId and original index
            new_df_raw['TransactionId'] = new_df_raw['CustomerId'] + '_' + new_df_raw.index.map(str)
        else:
            # Fallback if 'user_id' is missing (shouldn't happen with dummy data but good for robustness)
            new_df_raw['CustomerId'] = [f'CUST_{i}' for i in range(len(new_df_raw))]
            new_df_raw['TransactionId'] = [f'TXN_{i}' for i in range(len(new_df_raw))]
    elif dataset_type == 'creditcard':
        # For credit card, dummy data already has 'TransactionId' and 'CustomerId'
        # If not, create generic ones (though dummy data in main block ensures they exist)
        if 'TransactionId' not in new_df_raw.columns:
            new_df_raw['TransactionId'] = [f'CC_TXN_{i}' for i in range(len(new_df_raw))]
        if 'CustomerId' not in new_df_raw.columns:
            new_df_raw['CustomerId'] = [f'CC_CUST_{i}' for i in range(len(new_df_raw))]

    # Now, create X_new_raw_for_processing as a copy of the potentially modified new_df_raw
    X_new_raw_for_processing = new_df_raw.copy()

    # Apply initial transformations specific to the dataset type, mirroring run_train.py
    if dataset_type == 'ecommerce':
        # Rename columns for the processor's expected input schema
        X_new_raw_for_processing = X_new_raw_for_processing.rename(columns={
            'purchase_value': 'Amount',
            'purchase_time': 'TransactionStartTime'
        })

        # Handle missing 'signup_time' or 'ip_address' in new data if they are expected by processor
        if 'signup_time' not in X_new_raw_for_processing.columns:
            X_new_raw_for_processing['signup_time'] = pd.NaT # Use NaT for missing datetime
        if 'ip_address' not in X_new_raw_for_processing.columns:
            X_new_raw_for_processing['ip_address'] = np.nan # Use NaN for missing IP

        # Merge IP to country if the processor expects it and data is available
        ip_to_country_path = RAW_DATA_DIR / "IpAddress_to_Country.csv"
        ip_country_df_raw = load_data(ip_to_country_path)
        if not ip_country_df_raw.empty:
            X_new_raw_for_processing = merge_ip_to_country(X_new_raw_for_processing, ip_country_df_raw.copy())
        else:
            print(f"Warning: IP_TO_COUNTRY_PATH not found at {get_relative_path(ip_to_country_path)} for E-commerce data. Skipping IP merge.")
            if 'country' not in X_new_raw_for_processing.columns:
                X_new_raw_for_processing['country'] = 'Unknown' # Ensure 'country' column exists

    print("Processing new data using the loaded preprocessor...")
    try:
        # Use the loaded_processor to transform the new raw data
        X_predict_processed = loaded_processor.transform(X_new_raw_for_processing)
    except Exception as e:
        print(f"Error during preprocessing new data: {e}")
        print("Ensure the new data schema matches the training data schema expected by the processor.")
        return

    # Final safeguard: Ensure all features are numeric and handle any remaining NaNs
    non_numeric_cols = X_predict_processed.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found in processed new data: {non_numeric_cols}. Attempting to convert/drop.")
        for col in non_numeric_cols:
            try:
                X_predict_processed[col] = pd.to_numeric(X_predict_processed[col], errors='coerce').fillna(0.0)
            except Exception:
                print(f"Could not convert column {col} to numeric, dropping.")
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
    # For classification, we expect predict_proba
    if hasattr(loaded_model, 'predict_proba'):
        predictions_proba = loaded_model.predict_proba(X_predict_processed)[:, 1]
    else:
        # Fallback for models without predict_proba (e.g., some regression models, or if it's a binary classifier that only has predict)
        print("Warning: Model does not have 'predict_proba'. Using 'predict' method directly.")
        predictions_proba = loaded_model.predict(X_predict_processed)
        # If it's a binary classifier, ensure predictions_proba is still 0-1 range
        if np.issubdtype(predictions_proba.dtype, np.integer):
            # If predictions are binary (0 or 1), treat them as probabilities for simplicity (0.0 or 1.0)
            predictions_proba = predictions_proba.astype(float)


    risk_predictions = (predictions_proba >= 0.5).astype(int)

    # Add predictions to the original raw data DataFrame
    new_df_raw['predicted_risk_proba'] = pd.Series(predictions_proba, index=new_df_raw.index)
    new_df_raw['predicted_risk_label'] = pd.Series(risk_predictions, index=new_df_raw.index)


    print("\nPredictions complete. Sample of results:")
    # Display relevant columns, ensure they exist
    display_cols = ['TransactionId', 'CustomerId', 'Amount', 'predicted_risk_proba', 'predicted_risk_label']
    existing_display_cols = [col for col in display_cols if col in new_df_raw.columns]
    print(new_df_raw[existing_display_cols].head())

    predictions_output_path = PREDICTIONS_DIR / f"new_data_predictions_{dataset_type}.csv"
    new_df_raw.to_csv(predictions_output_path, index=False)
    print(f"\nPredictions saved to: {get_relative_path(predictions_output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fraud risk prediction on new data.")
    parser.add_argument('--generate-dummy-data', action='store_true',
                        help='Generate new dummy raw data files if they do not exist.')
    args = parser.parse_args()

    # Example usage for E-commerce Fraud data
    ecommerce_dummy_data_path = RAW_DATA_DIR / "new_ecommerce_transactions.csv"
    ip_country_data_path = RAW_DATA_DIR / "IpAddress_to_Country.csv" # Needed for dummy data creation

    if args.generate_dummy_data or not ecommerce_dummy_data_path.exists():
        print(f"Creating a dummy new E-commerce data file at {get_relative_path(ecommerce_dummy_data_path)} for demonstration.")
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
            'ip_address': ['100.0.0.1', '100.0.0.2', '100.0.0.3', '100.0.0.4'] # Changed to string IPs
        }
        pd.DataFrame(dummy_data_ecommerce).to_csv(ecommerce_dummy_data_path, index=False)
        print("Dummy E-commerce data created.")
    else:
        print(f"Using existing dummy E-commerce data at {get_relative_path(ecommerce_dummy_data_path)}.")

    # Run prediction for E-commerce data
    predict_risk(ecommerce_dummy_data_path, dataset_type='ecommerce')

    print("\n" + "="*50 + "\n")

    # Example usage for Credit Card Fraud data
    creditcard_dummy_data_path = RAW_DATA_DIR / "new_creditcard_transactions.csv"
    if args.generate_dummy_data or not creditcard_dummy_data_path.exists():
        print(f"Creating a dummy new Credit Card data file at {get_relative_path(creditcard_dummy_data_path)} for demonstration.")
        # Create dummy data matching creditcard.csv structure (V1-V28, Time, Amount)
        dummy_data_creditcard = {f'V{i}': np.random.rand(4) for i in range(1, 29)}
        dummy_data_creditcard['Time'] = [100, 200, 300, 400] # Dummy time
        dummy_data_creditcard['Amount'] = [50.0, 120.0, 30.0, 250.0] # Dummy amount
        # Add a dummy TransactionId and CustomerId for consistency with output, though not in original creditcard.csv
        dummy_data_creditcard['TransactionId'] = ['CC_T1', 'CC_T2', 'CC_T3', 'CC_T4']
        dummy_data_creditcard['CustomerId'] = ['CC_C1', 'CC_C2', 'CC_C3', 'CC_C4']
        pd.DataFrame(dummy_data_creditcard).to_csv(creditcard_dummy_data_path, index=False)
        print("Dummy Credit Card data created.")
    else:
        print(f"Using existing dummy Credit Card data at {get_relative_path(creditcard_dummy_data_path)}.")

    # Run prediction for Credit Card data
    predict_risk(creditcard_dummy_data_path, dataset_type='creditcard')
