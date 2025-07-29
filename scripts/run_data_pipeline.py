import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt # Keep for imbalance demo plots if any
import seaborn as sns # Keep for imbalance demo plots if any
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # For handling class imbalance

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import data loading and preprocessing components (now GPU-aware)
from src.data_processing.loader import load_data
from src.data_processing.preprocessor import FraudDataProcessor, CreditCardDataProcessor
from src.utils.helpers import merge_ip_to_country # Import the helper function from utils

# Attempt to import cudf for GPU acceleration
try:
    import cudf
    _CUDF_AVAILABLE = True
    print("cuDF is available in run_data_pipeline.py. GPU processing can be enabled.")
except ImportError:
    _CUDF_AVAILABLE = False
    print("cuDF not available in run_data_pipeline.py. GPU processing will be skipped.")


# --- Configuration and Data Paths ---
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed" # New directory for processed data

FRAUD_DATA_PATH = RAW_DATA_DIR / "Fraud_Data.csv"
IP_TO_COUNTRY_PATH = RAW_DATA_DIR / "IpAddress_to_Country.csv"
CREDITCARD_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

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
# All V features, Time, Amount are numerical. 'Time' and 'Amount' are explicitly handled.
CREDITCARD_NUMERICAL_FEATURES = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']


def load_all_data(use_gpu: bool):
    """Loads all necessary datasets, optionally using GPU."""
    print("\nLoading datasets...")
    # Explicitly define dtypes for IP columns to ensure they are loaded as strings
    ip_country_dtypes = {
        'lower_bound_ip_address': str,
        'upper_bound_ip_address': str
    }
    fraud_data_dtypes = {
        'ip_address': str
    }

    fraud_data_df = load_data(FRAUD_DATA_PATH, use_gpu=use_gpu, column_dtypes=fraud_data_dtypes)
    ip_country_df = load_data(IP_TO_COUNTRY_PATH, use_gpu=use_gpu, column_dtypes=ip_country_dtypes)
    creditcard_df = load_data(CREDITCARD_DATA_PATH, use_gpu=use_gpu) # No specific dtypes for creditcard.csv
    return fraud_data_df, ip_country_df, creditcard_df


def preprocess_fraud_data(fraud_df: pd.DataFrame, ip_country_df: pd.DataFrame, use_gpu: bool) -> pd.DataFrame:
    """Applies preprocessing steps specific to Fraud_Data.csv."""
    print("\n--- Processing E-commerce Fraud Data (Fraud_Data.csv) ---")
    if fraud_df.empty:
        print("Fraud_Data.csv is empty. Skipping preprocessing.")
        return pd.DataFrame()

    # Merge IP addresses to countries first
    # merge_ip_to_country is designed to handle cuDF internally by converting to pandas and back
    fraud_df = merge_ip_to_country(fraud_df, ip_country_df)

    # Separate features and target
    if FRAUD_TARGET_COL in fraud_df.columns:
        X_fraud = fraud_df.drop(columns=[FRAUD_TARGET_COL])
        y_fraud = fraud_df[FRAUD_TARGET_COL]
    else:
        print(f"Warning: Target column '{FRAUD_TARGET_COL}' not found in Fraud_Data.csv. Proceeding without target.")
        X_fraud = fraud_df.copy()
        # Create dummy target of the correct type (pandas or cudf Series)
        y_fraud = (cudf.Series([0] * len(X_fraud), index=X_fraud.index) if use_gpu and _CUDF_AVAILABLE
                   else pd.Series([0] * len(X_fraud), index=X_fraud.index))

    # Rename columns in X_fraud to match generic names expected by FraudDataProcessor
    X_fraud_renamed = X_fraud.rename(columns={
        'user_id': 'CustomerId',
        'purchase_value': 'Amount',
        'purchase_time': 'TransactionStartTime',
        'user_id': 'TransactionId' # Using user_id as a proxy for TransactionId for frequency count
    }).copy()
    
    fraud_numerical_features_renamed = ['Amount', 'age']
    fraud_categorical_features_renamed = ['source', 'browser', 'sex', 'country']
    fraud_purchase_time_col_renamed = 'TransactionStartTime'
    fraud_signup_time_col_renamed = 'signup_time'
    fraud_amount_col_renamed = 'Amount'
    fraud_id_cols_for_agg_renamed = ['CustomerId', 'device_id', 'ip_address']

    processor = FraudDataProcessor(
        numerical_cols_after_rename=fraud_numerical_features_renamed,
        categorical_cols_after_merge=fraud_categorical_features_renamed,
        time_col_after_rename=fraud_purchase_time_col_renamed,
        signup_time_col_after_rename=fraud_signup_time_col_renamed,
        amount_col_after_rename=fraud_amount_col_renamed,
        id_cols_for_agg_after_rename=fraud_id_cols_for_agg_renamed
    )

    print("Applying preprocessing pipeline to Fraud_Data.csv...")
    X_fraud_processed = processor.fit_transform(X_fraud_renamed, y_fraud)
    
    # Re-attach target for saving and potential later use
    # CRITICAL FIX: Conditionally use cudf.concat or pd.concat
    if use_gpu and _CUDF_AVAILABLE:
        # Ensure y_fraud is a cuDF Series if X_fraud_processed is cuDF
        if isinstance(y_fraud, pd.Series):
            y_fraud_for_concat = cudf.Series(y_fraud.values, index=y_fraud.index)
        else:
            y_fraud_for_concat = y_fraud
        fraud_processed_df = cudf.concat([X_fraud_processed, y_fraud_for_concat.rename(FRAUD_TARGET_COL)], axis=1)
    else:
        # Ensure y_fraud is a pandas Series if X_fraud_processed is pandas
        if isinstance(y_fraud, cudf.Series):
            y_fraud_for_concat = y_fraud.to_pandas()
        else:
            y_fraud_for_concat = y_fraud
        fraud_processed_df = pd.concat([X_fraud_processed, y_fraud_for_concat.rename(FRAUD_TARGET_COL)], axis=1)

    print("Fraud_Data.csv preprocessing complete.")
    return fraud_processed_df


def preprocess_creditcard_data(creditcard_df: pd.DataFrame, use_gpu: bool) -> pd.DataFrame:
    """Applies preprocessing steps specific to creditcard.csv."""
    print("\n--- Processing Bank Credit Card Fraud Data (creditcard.csv) ---")
    if creditcard_df.empty:
        print("creditcard.csv is empty. Skipping processing.")
        return pd.DataFrame()

    # Separate features and target
    if CREDITCARD_TARGET_COL in creditcard_df.columns:
        X_creditcard = creditcard_df.drop(columns=[CREDITCARD_TARGET_COL])
        y_creditcard = creditcard_df[CREDITCARD_TARGET_COL]
    else:
        print(f"Warning: Target column '{CREDITCARD_TARGET_COL}' not found in creditcard.csv. Proceeding without target.")
        X_creditcard = creditcard_df.copy()
        # Create dummy target of the correct type (pandas or cudf Series)
        y_creditcard = (cudf.Series([0] * len(X_creditcard), index=X_creditcard.index) if use_gpu and _CUDF_AVAILABLE
                        else pd.Series([0] * len(X_creditcard), index=X_creditcard.index))

    processor = CreditCardDataProcessor(
        numerical_cols=CREDITCARD_NUMERICAL_FEATURES
    )

    print("Applying preprocessing pipeline to creditcard.csv...")
    X_creditcard_processed = processor.fit_transform(X_creditcard, y_creditcard)
    
    # Re-attach target for saving and potential later use
    # CRITICAL FIX: Conditionally use cudf.concat or pd.concat
    if use_gpu and _CUDF_AVAILABLE:
        # Ensure y_creditcard is a cuDF Series if X_creditcard_processed is cuDF
        if isinstance(y_creditcard, pd.Series):
            y_creditcard_for_concat = cudf.Series(y_creditcard.values, index=y_creditcard.index)
        else:
            y_creditcard_for_concat = y_creditcard
        creditcard_processed_df = cudf.concat([X_creditcard_processed, y_creditcard_for_concat.rename(CREDITCARD_TARGET_COL)], axis=1)
    else:
        # Ensure y_creditcard is a pandas Series if X_creditcard_processed is pandas
        if isinstance(y_creditcard, cudf.Series):
            y_creditcard_for_concat = y_creditcard.to_pandas()
        else:
            y_creditcard_for_concat = y_creditcard
        creditcard_processed_df = pd.concat([X_creditcard_processed, y_creditcard_for_concat.rename(CREDITCARD_TARGET_COL)], axis=1)

    print("creditcard.csv preprocessing complete.")
    return creditcard_processed_df


def demonstrate_class_imbalance_handling(df: pd.DataFrame, dataset_name: str, target_col: str, use_gpu: bool):
    """
    Demonstrates class imbalance handling using SMOTE on a conceptual training set.
    This function remains in the data pipeline script as it's a data transformation step
    that prepares data for modeling, though typically applied during model training.
    """
    print(f"\n--- Demonstrating Class Imbalance Handling for {dataset_name} ---")
    if df.empty or target_col not in df.columns:
        print(f"Cannot demonstrate imbalance handling for empty or missing target column in {dataset_name}.")
        return

    # Convert to pandas for scikit-learn's train_test_split and SMOTE
    df_pd = df.to_pandas() if use_gpu and _CUDF_AVAILABLE and isinstance(df, cudf.DataFrame) else df.copy()

    X = df_pd.drop(columns=[target_col])
    y = df_pd[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"\nOriginal training set class distribution for {dataset_name}:")
    print(y_train.value_counts())
    print(f"Fraudulent transactions: {y_train.value_counts().get(1, 0)} ({y_train.value_counts().get(1, 0) / len(y_train) * 100:.2f}%) ")

    print(f"\nApplying SMOTE to the training data for {dataset_name}...")
    smote = SMOTE(random_state=42)
    
    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Resampled training set class distribution for {dataset_name}:")
        print(y_train_resampled.value_counts())
        print(f"Fraudulent transactions: {y_train_resampled.value_counts().get(1, 0)} ({y_train_resampled.value_counts().get(1, 0) / len(y_train_resampled) * 100:.2f}%) ")
        print("SMOTE applied successfully.")
    except Exception as e:
        print(f"Error applying SMOTE: {e}")
        print("Ensure all features are numerical (float, int) before applying SMOTE.")
        print("Check dtypes of X_train:")
        print(X_train.dtypes[X_train.dtypes == 'object'])

    print(f"\n--- Class Imbalance Handling Demonstration for {dataset_name} Complete ---")


def run_data_pipeline_main(use_gpu: bool = False):
    """
    Orchestrates the data loading, preprocessing, feature engineering,
    and saving of processed data for both datasets, optionally using GPU.
    """
    print("--- Starting Data Engineering Pipeline ---")
    if use_gpu and not _CUDF_AVAILABLE:
        print("Warning: GPU processing requested, but cuDF is not available. Falling back to CPU.")
        use_gpu = False

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fraud_data_df, ip_country_df, creditcard_df = load_all_data(use_gpu=use_gpu)

    fraud_processed_df = preprocess_fraud_data(fraud_data_df.copy(), ip_country_df.copy(), use_gpu=use_gpu)

    creditcard_processed_df = preprocess_creditcard_data(creditcard_df.copy(), use_gpu=use_gpu)

    # --- Save Processed Data ---
    # Convert back to pandas before saving to CSV for broader compatibility
    if use_gpu and _CUDF_AVAILABLE:
        if not fraud_processed_df.empty:
            fraud_output_path = PROCESSED_DATA_DIR / "fraud_processed.csv"
            fraud_processed_df.to_pandas().to_csv(fraud_output_path, index=False)
            print(f"\nProcessed Fraud_Data.csv saved to: {fraud_output_path}")
        else:
            print("\nNo processed Fraud_Data.csv to save.")

        if not creditcard_processed_df.empty:
            creditcard_output_path = PROCESSED_DATA_DIR / "creditcard_processed.csv"
            creditcard_processed_df.to_pandas().to_csv(creditcard_output_path, index=False)
            print(f"\nProcessed creditcard.csv saved to: {creditcard_output_path}")
        else:
            print("\nNo processed creditcard.csv to save.")
    else: # Pandas path
        if not fraud_processed_df.empty:
            fraud_output_path = PROCESSED_DATA_DIR / "fraud_processed.csv"
            fraud_processed_df.to_csv(fraud_output_path, index=False)
            print(f"\nProcessed Fraud_Data.csv saved to: {fraud_output_path}")
        else:
            print("\nNo processed Fraud_Data.csv to save.")

        if not creditcard_processed_df.empty:
            creditcard_output_path = PROCESSED_DATA_DIR / "creditcard_processed.csv"
            creditcard_processed_df.to_csv(creditcard_output_path, index=False)
            print(f"\nProcessed creditcard.csv saved to: {creditcard_output_path}")
        else:
            print("\nNo processed creditcard.csv to save.")
    # --- End Save Processed Data ---

    demonstrate_class_imbalance_handling(creditcard_processed_df.copy(), "Bank Credit Card Fraud Data (creditcard.csv)", CREDITCARD_TARGET_COL, use_gpu=use_gpu)
    demonstrate_class_imbalance_handling(fraud_processed_df.copy(), "E-commerce Fraud Data (Fraud_Data.csv)", FRAUD_TARGET_COL, use_gpu=use_gpu)


    print("\n--- Data Engineering Pipeline Complete ---")


if __name__ == "__main__":
    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' not found. Please ensure 'data/raw' exists and contains the datasets.")
        sys.exit(1)

    required_raw_files = [FRAUD_DATA_PATH, IP_TO_COUNTRY_PATH, CREDITCARD_DATA_PATH]
    for file_path in required_raw_files:
        if not file_path.exists():
            print(f"Error: Required raw data file '{file_path.name}' not found in '{RAW_DATA_DIR}'. Please place it there.")
            sys.exit(1)

    # Set use_gpu=True to enable GPU acceleration in Colab
    run_data_pipeline_main(use_gpu=True)
