import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # For handling class imbalance

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import data loading and preprocessing components
from src.data_processing.loader import load_data, get_relative_path # Import get_relative_path
from src.data_processing.preprocessor import FraudDataProcessor, CreditCardDataProcessor
from src.utils.helpers import merge_ip_to_country # Import the helper function from utils


# --- Configuration and Data Paths ---
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed" # New directory for processed data

FRAUD_DATA_PATH = RAW_DATA_DIR / "Fraud_Data.csv"
IP_TO_COUNTRY_PATH = RAW_DATA_DIR / "IpAddress_to_Country.csv"
CREDITCARD_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

# Ensure processed data directory exists
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- Column Definitions for Fraud_Data.csv ---
FRAUD_TARGET_COL = 'class'
FRAUD_NUMERICAL_FEATURES = ['purchase_value', 'age']
FRAUD_CATEGORICAL_FEATURES = ['source', 'browser', 'sex', 'country'] # 'country' is added after merge
FRAUD_PURCHASE_TIME_COL = 'purchase_time'
FRAUD_SIGNUP_TIME_COL = 'signup_time'
FRAUD_AMOUNT_COL = 'purchase_value' # The column to treat as 'Amount' for HandleAmountFeatures
FRAUD_ID_COLS_FOR_AGG = ['user_id', 'device_id', 'ip_address'] # IDs for frequency/velocity

# --- Column Definitions for creditcard.csv ---
CREDITCARD_TARGET_COL = 'Class'
CREDITCARD_NUMERICAL_FEATURES = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']


def demonstrate_class_imbalance_handling(df: pd.DataFrame, dataset_name: str, target_col: str):
    """
    Demonstrates class imbalance handling using SMOTE.
    This is for demonstration purposes within the data pipeline script.
    """
    if df.empty or target_col not in df.columns:
        print(f"Skipping imbalance demonstration for {dataset_name}: DataFrame is empty or target column '{target_col}' not found.")
        return

    print(f"\n--- Class Imbalance Handling Demonstration for {dataset_name} ---")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Original target distribution:\n{y.value_counts(normalize=True)}")

    # Check for NaNs before SMOTE
    if X.isnull().any().any():
        print(f"WARNING: NaNs detected in features for {dataset_name} before SMOTE. Imputing with 0 for demonstration.")
        X = X.fillna(0) # Fill NaNs for demonstration purposes

    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        print(f"Resampled data shape: {X_resampled.shape}")
        print(f"Resampled target distribution:\n{y_resampled.value_counts(normalize=True)}")
    except Exception as e:
        print(f"Error applying SMOTE for {dataset_name} demonstration: {e}")
        print("This might be due to remaining NaNs or a single class in the data.")


def run_data_engineering_pipeline():
    """
    Executes the full data engineering pipeline for both datasets:
    1. Loads raw data.
    2. Merges IP to Country data (for Fraud_Data.csv).
    3. Initializes and applies specific data processors.
    4. Saves processed data to 'data/processed' directory.
    5. Demonstrates class imbalance handling.
    """
    print("--- Starting Data Engineering Pipeline ---")

    # --- 1. Load Raw Data ---
    print("\nLoading raw data...")
    fraud_df_raw = load_data(FRAUD_DATA_PATH, column_dtypes={'ip_address': str})
    ip_country_df_raw = load_data(IP_TO_COUNTRY_PATH, column_dtypes={'lower_bound_ip_address': str, 'upper_bound_ip_address': str})
    creditcard_df_raw = load_data(CREDITCARD_DATA_PATH)

    if fraud_df_raw.empty or ip_country_df_raw.empty:
        print("Warning: E-commerce Fraud data or IP country data not loaded. Skipping E-commerce Fraud processing.")
        fraud_processed_df = pd.DataFrame()
    else:
        print("\n--- Processing E-commerce Fraud Data (Fraud_Data.csv) ---")
        # Apply deduplication to raw data before processing
        initial_fraud_rows = fraud_df_raw.shape[0]
        fraud_df_deduplicated = fraud_df_raw.drop_duplicates().copy()
        fraud_rows_removed = initial_fraud_rows - fraud_df_deduplicated.shape[0]
        if fraud_rows_removed > 0:
            print(f"Removed {fraud_rows_removed} duplicate rows from raw Fraud_Data.csv.")
        else:
            print("No duplicate rows found in raw Fraud_Data.csv.")

        # Separate features and target from deduplicated raw data
        X_fraud = fraud_df_deduplicated.drop(columns=[FRAUD_TARGET_COL]).copy()
        y_fraud = fraud_df_deduplicated[FRAUD_TARGET_COL].copy().astype(int)

        # Merge IP addresses to countries (this modifies X_fraud in place or returns new df)
        # Ensure column names are consistent with what FraudDataProcessor expects after rename
        X_fraud_pre_process = X_fraud.rename(columns={
            'user_id': 'CustomerId',
            'purchase_value': 'Amount',
            'purchase_time': 'TransactionStartTime'
        }).copy()
        X_fraud_merged = merge_ip_to_country(X_fraud_pre_process, ip_country_df_raw.copy())
        
        # Ensure TransactionId is present for frequency velocity calculations if 'user_id' was used
        if 'user_id' in fraud_df_deduplicated.columns:
            X_fraud_merged['TransactionId'] = fraud_df_deduplicated['user_id'] # Use deduplicated df here


        fraud_processor = FraudDataProcessor(
            numerical_cols_after_rename=FRAUD_NUMERICAL_FEATURES,
            categorical_cols_after_merge=FRAUD_CATEGORICAL_FEATURES,
            time_col_after_rename='TransactionStartTime',
            signup_time_col_after_rename='signup_time',
            amount_col_after_rename='Amount',
            id_cols_for_agg_after_rename=['CustomerId', 'device_id', 'ip_address']
        )
        print("Applying FraudDataProcessor pipeline...")
        fraud_processed_df = fraud_processor.fit_transform(X_fraud_merged, y_fraud)

        # --- NEW: Final NaN check and imputation for fraud_processed_df ---
        if fraud_processed_df.isnull().any().any():
            print("WARNING: NaNs detected in processed E-commerce Fraud features AFTER pipeline. Imputing with 0.")
            for col in fraud_processed_df.columns:
                if fraud_processed_df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(fraud_processed_df[col]):
                        fraud_processed_df[col] = fraud_processed_df[col].fillna(0.0)
                    else:
                        fraud_processed_df[col] = fraud_processed_df[col].fillna('missing_value')
        # --- END NEW ---

        # Concatenate features and target for saving
        fraud_processed_df[FRAUD_TARGET_COL] = y_fraud.values # Ensure target is aligned and added back


    if creditcard_df_raw.empty:
        print("Warning: Credit Card Fraud data not loaded. Skipping Credit Card Fraud processing.")
        creditcard_processed_df = pd.DataFrame()
    else:
        print("\n--- Processing Bank Credit Card Fraud Data (creditcard.csv) ---")
        # Apply deduplication to raw data before processing
        initial_creditcard_rows = creditcard_df_raw.shape[0]
        creditcard_df_deduplicated = creditcard_df_raw.drop_duplicates().copy()
        creditcard_rows_removed = initial_creditcard_rows - creditcard_df_deduplicated.shape[0]
        if creditcard_rows_removed > 0:
            print(f"Removed {creditcard_rows_removed} duplicate rows from raw creditcard.csv.")
        else:
            print("No duplicate rows found in raw creditcard.csv.")

        # Separate features and target from deduplicated raw data
        X_creditcard = creditcard_df_deduplicated.drop(columns=[CREDITCARD_TARGET_COL]).copy()
        y_creditcard = creditcard_df_deduplicated[CREDITCARD_TARGET_COL].copy().astype(int)

        creditcard_processor = CreditCardDataProcessor(
            numerical_cols=CREDITCARD_NUMERICAL_FEATURES
        )
        print("Applying CreditCardDataProcessor pipeline...")
        creditcard_processed_df = creditcard_processor.fit_transform(X_creditcard, y_creditcard)

        # --- NEW: Final NaN check and imputation for creditcard_processed_df ---
        if creditcard_processed_df.isnull().any().any():
            print("WARNING: NaNs detected in processed Credit Card Fraud features AFTER pipeline. Imputing with 0.")
            for col in creditcard_processed_df.columns:
                if creditcard_processed_df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(creditcard_processed_df[col]):
                        creditcard_processed_df[col] = creditcard_processed_df[col].fillna(0.0)
                    else:
                        creditcard_processed_df[col] = creditcard_processed_df[col].fillna('missing_value')
        # --- END NEW ---

        # Concatenate features and target for saving
        creditcard_processed_df[CREDITCARD_TARGET_COL] = y_creditcard.values # Ensure target is aligned and added back


    # --- Save Processed Data ---
    print("\nSaving processed data...")
    if not fraud_processed_df.empty:
        fraud_output_path = PROCESSED_DATA_DIR / "fraud_processed.csv"
        fraud_processed_df.to_csv(fraud_output_path, index=False)
        print(f"\nProcessed Fraud_Data.csv saved to: {get_relative_path(fraud_output_path)}")
    else:
        print("\nNo processed Fraud_Data.csv to save.")

    if not creditcard_processed_df.empty:
        creditcard_output_path = PROCESSED_DATA_DIR / "creditcard_processed.csv"
        creditcard_processed_df.to_csv(creditcard_output_path, index=False)
        print(f"\nProcessed creditcard.csv saved to: {get_relative_path(creditcard_output_path)}")
    else:
        print("\nNo processed creditcard.csv to save.")
    # --- End Save Processed Data ---

    # Demonstrate imbalance handling using the *saved* processed data (or the in-memory if not saved)
    # Load them again to ensure we're using the same data that run_train.py would load
    print("\n--- Re-loading processed data for imbalance demonstration ---")
    reloaded_fraud_df = load_data(PROCESSED_DATA_DIR / "fraud_processed.csv")
    reloaded_creditcard_df = load_data(PROCESSED_DATA_DIR / "creditcard_processed.csv")

    demonstrate_class_imbalance_handling(reloaded_creditcard_df.copy(), "Bank Credit Card Fraud Data (creditcard.csv)", CREDITCARD_TARGET_COL)
    demonstrate_class_imbalance_handling(reloaded_fraud_df.copy(), "E-commerce Fraud Data (Fraud_Data.csv)", FRAUD_TARGET_COL)


    print("\n--- Data Engineering Pipeline Complete ---")


if __name__ == "__main__":
    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' not found. Please ensure 'data/raw' exists and contains the datasets.")
        sys.exit(1)

    required_raw_files = [FRAUD_DATA_PATH, IP_TO_COUNTRY_PATH, CREDITCARD_DATA_PATH]
    for file_path in required_raw_files:
        if not file_path.exists():
            print(f"Error: Required raw data file '{get_relative_path(file_path)}' not found. Please ensure it exists in '{get_relative_path(RAW_DATA_DIR)}'.")
            sys.exit(1)
            
    run_data_engineering_pipeline()

