import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cloudpickle # Needed for loading processor
import argparse

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Fix import path for load_data
from src.data_processing.loader import load_data, get_relative_path
from src.data_processing.preprocessor import FraudDataProcessor, CreditCardDataProcessor
from src.models.model_interpreter import ModelInterpreter
from src.utils.helpers import merge_ip_to_country


# Define paths
MLRUNS_PATH = project_root / "mlruns" # MLflow tracking directory
RAW_DATA_DIR = project_root / "data" / "raw" # Define raw data directory for consistency
EXPORTED_MODELS_DIR = project_root / "exported_models" # NEW: Define exported models directory

# Ensure MLflow tracking URI is set
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


def run_model_interpretation(
    dataset_path: Path,
    dataset_type: str, # 'ecommerce' or 'creditcard'
    registered_model_name: str,
    model_version: str = "latest",
    ip_country_path: Path = None, # Optional for e-commerce data
    num_samples_for_shap: int = 500, # Number of samples for global SHAP explanation
    num_instances_for_lime: int = 5 # Number of individual instances for LIME explanation
):
    """
    Loads a registered model and its associated data processor,
    preprocesses new data, and performs model interpretation using SHAP and LIME.

    Args:
        dataset_path (Path): Path to the raw data CSV file for interpretation.
        dataset_type (str): Type of dataset ('ecommerce' or 'creditcard').
        registered_model_name (str): The name of the registered model in MLflow.
        model_version (str): The version of the model to load (e.g., '1', 'latest').
        ip_country_path (Path, optional): Path to the IP to Country CSV file, required for 'ecommerce' type.
        num_samples_for_shap (int): Number of samples to use for global SHAP explanation.
        num_instances_for_lime (int): Number of individual instances for LIME explanation.
    """
    print(f"Starting model interpretation for '{dataset_type}' data using model '{registered_model_name}' version '{model_version}'...")

    # --- 1. Load Raw Data for Interpretation ---
    raw_df = load_data(dataset_path)
    if raw_df.empty:
        print(f"Error: Interpretation data from {get_relative_path(dataset_path)} is empty. Exiting.")
        return

    # --- 2. Load Model and Processor ---
    sklearn_loaded_model = None
    loaded_processor = None
    client = mlflow.tracking.MlflowClient()

    try:
        # Get the latest version of the model
        latest_versions = client.get_latest_versions(registered_model_name, stages=["None", "Staging", "Production"])

        model_info = None
        if model_version == "latest":
            if latest_versions:
                model_info = latest_versions[0]
        else:
            for mv in latest_versions:
                if str(mv.version) == model_version:
                    model_info = mv
                    break

        if model_info is None:
            raise ValueError(f"Model '{registered_model_name}' version '{model_version}' not found in MLflow Registry.")

        model_uri = f"models:/{registered_model_name}/{model_info.version}"
        print(f"Loading model from URI: {model_uri}")
        sklearn_loaded_model = mlflow.sklearn.load_model(model_uri)

        # Attempt to load the processor from MLflow artifact first
        run_id_of_registered_model = model_info.run_id
        processor_artifact_uri = f"runs:/{run_id_of_registered_model}/processor"
        print(f"Attempting to load processor from MLflow artifact URI: {processor_artifact_uri}")
        
        try:
            local_processor_path_mlflow = client.download_artifacts(run_id=run_id_of_registered_model, path="processor")
            with open(local_processor_path_mlflow, 'rb') as f:
                loaded_processor = cloudpickle.load(f)
            print("Successfully loaded processor from MLflow artifact.")
        except Exception as e_mlflow_download:
            print(f"Failed to download processor from MLflow artifact: {e_mlflow_download}")
            print("Attempting to load processor from locally exported .pkl file as a fallback.")

            # Fallback to local exported processor
            processor_file_name = f"best_{dataset_type}_processor.pkl"
            local_exported_processor_path = EXPORTED_MODELS_DIR / processor_file_name

            if local_exported_processor_path.exists():
                with open(local_exported_processor_path, 'rb') as f:
                    loaded_processor = cloudpickle.load(f)
                print(f"Successfully loaded processor from local file: {local_exported_processor_path}")
            else:
                raise FileNotFoundError(f"Local processor file not found at {local_exported_processor_path}. "
                                        "Please ensure run_train.py has successfully exported the processor.")

        print("Successfully loaded model and processor.")

    except Exception as e:
        print(f"Error loading model or processor for interpretation: {e}")
        print("Please ensure the model and its processor are registered (in MLflow) or exported (locally), and the MLflow tracking server is accessible.")
        return

    # --- 3. Preprocess the New Data using the Loaded Processor ---
    print(f"\n--- Preprocessing new data for interpretation ({dataset_type}) ---")

    initial_rows = raw_df.shape[0]
    df_deduplicated = raw_df.drop_duplicates().copy()
    rows_removed = initial_rows - df_deduplicated.shape[0]
    if rows_removed > 0:
        print(f"Removed {rows_removed} duplicate rows from raw interpretation data.")
    else:
        print("No duplicate rows found in raw interpretation data.")

    target_col_to_drop = CREDITCARD_TARGET_COL if dataset_type == 'creditcard' else FRAUD_TARGET_COL
    X_interpret_raw = df_deduplicated.drop(columns=[target_col_to_drop], errors='ignore').copy()

    if dataset_type == 'ecommerce':
        if ip_country_path and ip_country_path.exists():
            ip_country_df = load_data(ip_country_path)
            if ip_country_df.empty:
                print("Warning: IP to Country data is empty for E-commerce Fraud data processing.")

            X_interpret_raw = X_interpret_raw.rename(columns={
                'user_id': 'CustomerId',
                'purchase_value': 'Amount',
                'purchase_time': 'TransactionStartTime'
            }).copy()
            X_interpret_raw = merge_ip_to_country(X_interpret_raw, ip_country_df.copy())
            if 'user_id' in df_deduplicated.columns:
                X_interpret_raw['TransactionId'] = df_deduplicated['user_id']
        else:
            print("Warning: IP to Country path not provided or does not exist for E-commerce data, merge will be skipped.")
            X_interpret_raw = X_interpret_raw.rename(columns={
                'user_id': 'CustomerId',
                'purchase_value': 'Amount',
                'purchase_time': 'TransactionStartTime'
            }, errors='ignore').copy()
            if 'user_id' in df_deduplicated.columns:
                X_interpret_raw['TransactionId'] = df_deduplicated['user_id']

    # Transform using the loaded processor
    try:
        X_processed = loaded_processor.transform(X_interpret_raw)
    except Exception as e:
        print(f"Error transforming data with loaded processor: {e}")
        print("This might happen if the processed data columns from run_train.py are different from expected by the processor.")
        return

    if X_processed.isnull().any().any():
        print("WARNING: NaNs detected in processed features for interpretation. Imputing with 0 or 'missing_value'.")
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    X_processed[col] = X_processed[col].fillna(0.0)
                else:
                    X_processed[col] = X_processed[col].fillna('missing_value')

    print(f"Processed data shape for interpretation: {X_processed.shape}")

    # --- 4. Initialize Model Interpreter ---
    X_shap_background = X_processed.sample(n=min(num_samples_for_shap, len(X_processed)), random_state=42)
    num_samples_for_lime_actual = min(num_instances_for_lime, len(X_processed))
    X_lime_training_data = X_processed.sample(n=num_samples_for_lime_actual, random_state=42).values
    class_names = ['Low Risk', 'High Risk']

    interpreter = ModelInterpreter(
        model=sklearn_loaded_model,
        feature_names=X_processed.columns.tolist(),
        model_type='classification',
        class_names=class_names,
        training_data_for_lime=X_lime_training_data,
        max_background_samples_shap=num_samples_for_shap,
        max_background_samples_lime=num_samples_for_lime_actual
    )

    # --- 5. Global Interpretation (SHAP) ---
    print("\n--- Performing Global Interpretation with SHAP ---")
    print(f"Generating SHAP explanations for {len(X_shap_background)} samples...")
    interpreter.explain_model_shap(X_shap_background)
    plt.show() # Ensure plot is displayed
    interpreter.plot_shap_summary(X_shap_background) # Call plot_shap_summary explicitly
    plt.show() # Ensure plot is displayed

    # --- 6. Local Interpretation (LIME) ---
    print(f"\n--- Performing Local Interpretation with LIME for {num_instances_for_lime} instances ---")
    for i in tqdm(range(min(num_instances_for_lime, len(X_processed))), desc="LIME Explanations"):
        instance_to_explain = X_processed.iloc[i]
        print(f"\nLIME Explanation for instance {i+1} (Original Index: {instance_to_explain.name}):")
        interpreter.explain_instance_lime(instance_to_explain.values)
        plt.show() # Ensure plot is displayed

    print("\nModel interpretation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fraud detection model interpretation pipeline.")
    parser.add_argument('--generate-dummy-data', action='store_true',
                        help='Generate dummy interpretation data files if they do not exist.')
    args = parser.parse_args()

    # Define paths for raw data
    FRAUD_RAW_DATA_PATH = RAW_DATA_DIR / "Fraud_Data.csv"
    IP_TO_COUNTRY_RAW_DATA_PATH = RAW_DATA_DIR / "IpAddress_to_Country.csv"
    CREDITCARD_RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

    client = mlflow.tracking.MlflowClient()

    # --- Dynamically determine E-commerce Fraud model name ---
    ecommerce_model_name = None
    try:
        registered_models = client.search_registered_models(filter_string="name LIKE 'FraudDetection_%'")
        if registered_models:
            latest_model = max(registered_models, key=lambda m: m.creation_timestamp)
            ecommerce_model_name = latest_model.name
            print(f"Dynamically determined E-commerce model name: {ecommerce_model_name}")
        else:
            print("No registered E-commerce Fraud models found. Please run run_train.py first.")
    except Exception as e:
        print(f"Error searching for E-commerce Fraud models: {e}")

    # --- Dynamically determine Credit Card Fraud model name ---
    creditcard_model_name = None
    try:
        registered_models = client.search_registered_models(filter_string="name LIKE 'CreditCardFraud_%'")
        if registered_models:
            latest_model = max(registered_models, key=lambda m: m.creation_timestamp)
            creditcard_model_name = latest_model.name
            print(f"Dynamically determined Credit Card model name: {creditcard_model_name}")
        else:
            print("No registered Credit Card Fraud models found. Please run run_train.py first.")
    except Exception as e:
        print(f"Error searching for Credit Card Fraud models: {e}")


    # Example usage for E-commerce Fraud data
    ecommerce_dummy_interpret_data_path = RAW_DATA_DIR / "dummy_fraud_interpret_data.csv"
    if args.generate_dummy_data or not ecommerce_dummy_interpret_data_path.exists():
        print(f"Creating a dummy E-commerce interpretation data file at {get_relative_path(ecommerce_dummy_interpret_data_path)} for demonstration.")
        dummy_data_ecommerce_interpret = {
            'user_id': [1000000000, 1000000001, 1000000002, 1000000003, 1000000004],
            'signup_time': ['2019-01-01 12:00:00', '2019-01-02 13:00:00', '2019-01-03 14:00:00', '2019-01-04 15:00:00', '2019-01-05 16:00:00'],
            'purchase_time': ['2019-01-01 12:05:00', '2019-01-02 13:10:00', '2019-01-03 14:15:00', '2019-01-04 15:20:00', '2019-01-05 16:25:00'],
            'purchase_value': [50.0, 100.0, 20.0, 150.0, 80.0],
            'device_id': ['D1', 'D2', 'D3', 'D4', 'D5'],
            'source': ['SEO', 'Ads', 'Direct', 'SEO', 'Ads'],
            'browser': ['Chrome', 'Firefox', 'Safari', 'Edge', 'Chrome'],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'ip_address': ['192.168.1.1', '10.0.0.1', '172.16.0.1', '2.0.0.1', '192.168.1.2'],
            'class': [0, 1, 0, 1, 0]
        }
        pd.DataFrame(dummy_data_ecommerce_interpret).to_csv(ecommerce_dummy_interpret_data_path, index=False)
        print("Dummy E-commerce interpretation data created.")
    else:
        print(f"Using existing dummy E-commerce interpretation data at {get_relative_path(ecommerce_dummy_interpret_data_path)}.")

    if ecommerce_model_name:
        print(f"\n--- Running Interpretation for E-commerce Fraud Data ---")
        run_model_interpretation(
            dataset_path=ecommerce_dummy_interpret_data_path,
            ip_country_path=IP_TO_COUNTRY_RAW_DATA_PATH,
            dataset_type='ecommerce',
            registered_model_name=ecommerce_model_name,
            model_version="latest",
            num_samples_for_shap=1000,
            num_instances_for_lime=3
        )
    else:
        print("\nSkipping E-commerce Fraud interpretation as no registered model was found.")


    print("\n" + "="*50 + "\n")

    # Example usage for Credit Card Fraud data
    creditcard_dummy_interpret_data_path = RAW_DATA_DIR / "dummy_creditcard_interpret_data.csv"
    if args.generate_dummy_data or not creditcard_dummy_interpret_data_path.exists():
        print(f"Creating a dummy Credit Card interpretation data file at {get_relative_path(creditcard_dummy_interpret_data_path)} for demonstration.")
        dummy_data_creditcard_interpret = {f'V{i}': np.random.rand(5) for i in range(1, 29)}
        dummy_data_creditcard_interpret['Time'] = [500, 600, 700, 800, 900]
        dummy_data_creditcard_interpret['Amount'] = [60.0, 130.0, 40.0, 280.0, 95.0]
        dummy_data_creditcard_interpret['Class'] = [0, 1, 0, 1, 0]
        pd.DataFrame(dummy_data_creditcard_interpret).to_csv(creditcard_dummy_interpret_data_path, index=False)
        print("Dummy Credit Card interpretation data created.")
    else:
        print(f"Using existing dummy Credit Card interpretation data at {get_relative_path(creditcard_dummy_interpret_data_path)}.")

    if creditcard_model_name:
        print(f"\n--- Running Interpretation for Bank Credit Card Fraud Data ---")
        run_model_interpretation(
            dataset_path=creditcard_dummy_interpret_data_path,
            ip_country_path=None,
            dataset_type='creditcard',
            registered_model_name=creditcard_model_name,
            model_version="latest",
            num_samples_for_shap=1000,
            num_instances_for_lime=3
        )
    else:
        print("\nSkipping Credit Card Fraud interpretation as no registered model was found.")

    print("\nInterpretation process complete.")

