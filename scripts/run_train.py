import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
from pathlib import Path
import sys
import json
from tqdm import tqdm 
import os 
import cloudpickle
import argparse

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_processing.loader import load_data, get_relative_path
from src.data_processing.preprocessor import FraudDataProcessor, CreditCardDataProcessor
from src.models.model_trainer import ModelTrainer
from src.models.logistic_regression_strategy import LogisticRegressionStrategy
from src.models.decision_tree_strategy import DecisionTreeStrategy
from src.models.random_forest_strategy import RandomForestStrategy
from src.models.xgboost_strategy import XGBoostStrategy
# from src.models.lightgbm_strategy import LightGBMStrategy # REMOVED: LightGBM import
from src.models.model_evaluator import evaluate_classification_model
from src.utils.helpers import merge_ip_to_country

# Define paths
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
MLRUNS_PATH = project_root / "mlruns"
EXPORT_DIR = project_root / "exported_models" # This is the consistent variable name

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_PATH.mkdir(parents=True, exist_ok=True) 
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Set MLflow tracking URI to use local file system for experiment tracking
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")
mlflow.set_experiment("Fraud Detection Model Training")

# Global Random State for reproducibility
RANDOM_STATE = 42


# --- Column Definitions for Fraud_Data.csv ---
FRAUD_TARGET_COL = 'class'
FRAUD_NUMERICAL_FEATURES = ['purchase_value', 'age']
FRAUD_CATEGORICAL_FEATURES = ['source', 'browser', 'sex', 'country']
FRAUD_PURCHASE_TIME_COL = 'purchase_time'
FRAUD_SIGNUP_TIME_COL = 'signup_time'
FRAUD_AMOUNT_COL = 'purchase_value'
FRAUD_ID_COLS_FOR_AGG = ['user_id', 'device_id', 'ip_address']

# --- Column Definitions for creditcard.csv ---
CREDITCARD_TARGET_COL = 'Class'
CREDITCARD_NUMERICAL_FEATURES = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']


def train_and_log_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: object, # The fitted preprocessor
    model_strategy: object,
    experiment_name: str
):
    """
    Trains a model using the provided strategy, evaluates it, and logs everything to MLflow.
    
    Args:
        model_name (str): Name of the model (e.g., 'Logistic Regression', 'XGBoost Classifier').
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        preprocessor (object): The fitted data preprocessor.
        model_strategy (object): An instance of a model strategy (e.g., LogisticRegressionStrategy).
        experiment_name (str): Name of the MLflow experiment.
    
    Returns:
        dict: Evaluation metrics.
        
    """
    print(f"\n--- Training {model_name} for {experiment_name} ---")
    
    # Use mlflow.start_run as a context manager to ensure the run is properly ended
    with mlflow.start_run(run_name=f"{model_name}_Training", nested=True) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_preprocessing_strategy", "SMOTE_and_FeatureEngineering")
        
        trainer = ModelTrainer(model_strategy)
        trainer.train_model(X_train, y_train)
        
        # Make predictions and evaluate
        y_pred_proba = trainer.predict_model(X_test)
        metrics = evaluate_classification_model(y_test.values, y_pred_proba)
        
        # Log metrics
        print(f"Metrics for {model_name}:")
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            print(f"  {metric_name}: {value:.4f}")

        # Log the model to MLflow Registry
        mlflow.sklearn.log_model(
            sk_model=trainer.get_current_model_object(),
            artifact_path="model", # This is the artifact path *within the run*
            registered_model_name=f"{experiment_name}_{model_name.replace(' ', '')}" # This is the name in the MLflow Model Registry
        )

        # Log the preprocessor as a Python artifact (using cloudpickle for generality)
        # This logs it *within the current MLflow run's artifacts*
        # We'll also export it locally later.
        local_processor_artifact_path = EXPORT_DIR / f"{experiment_name}_{model_name.replace(' ', '')}_processor_mlflow_artifact.pkl"
        with open(local_processor_artifact_path, 'wb') as f:
            cloudpickle.dump(preprocessor, f)
        mlflow.log_artifact(local_path=str(local_processor_artifact_path), artifact_path="processor") # Log to MLflow
        # os.remove(local_processor_artifact_path) # <--- THIS LINE IS COMMENTED OUT NOW!


        print(f"Model '{model_name}' for {experiment_name} logged to MLflow run.")
        print(f"Processor for '{model_name}' for {experiment_name} logged as MLflow artifact.")

        return metrics, trainer.get_current_model_object() # Return the trained model object as well

# Moved this function definition before main()
def run_training_pipeline(
    fraud_raw_data_path: Path,
    ip_to_country_raw_data_path: Path,
    creditcard_raw_data_path: Path,
    skip_preprocessing: bool = False
):
    """
    Orchestrates the entire training pipeline for both fraud datasets.
    """
    print("Starting comprehensive model training and tracking process...")

    # --- 1. Load Data ---
    print("\n--- Loading Raw Datasets ---")
    fraud_raw_df = load_data(fraud_raw_data_path)
    ip_country_df = load_data(ip_to_country_raw_data_path)
    creditcard_raw_df = load_data(creditcard_raw_data_path)

    if fraud_raw_df.empty or ip_country_df.empty or creditcard_raw_df.empty:
        print("Error: One or more raw data files are empty. Exiting training pipeline.")
        return

    # --- 2. Preprocess Data if not skipped ---
    fraud_processed_df = pd.DataFrame()
    creditcard_processed_df = pd.DataFrame()
    fraud_processor = None
    creditcard_processor = None

    # Flag to track if preprocessing was actually skipped and successful
    preprocessing_skipped_successfully = False

    if skip_preprocessing:
        print("\n--- Attempting to skip Data Preprocessing (using existing processed files) ---")
        try:
            fraud_processed_df = load_data(PROCESSED_DATA_DIR / "fraud_processed.csv")
            creditcard_processed_df = load_data(PROCESSED_DATA_DIR / "creditcard_processed.csv")
            
            # Load processors from local exported files if skipping preprocessing
            fraud_processor_path = EXPORT_DIR / "best_ecommerce_processor.pkl"
            if fraud_processor_path.exists():
                with open(fraud_processor_path, 'rb') as f:
                    fraud_processor = cloudpickle.load(f)
                print(f"Loaded E-commerce Fraud processor from {fraud_processor_path}")
            else:
                print(f"Warning: E-commerce Fraud processor not found at {fraud_processor_path}. Will force full preprocessing for E-commerce data.")
                fraud_processor = None # Force full preprocessing
                fraud_processed_df = pd.DataFrame() # Clear to force reprocessing

            creditcard_processor_path = EXPORT_DIR / "best_creditcard_processor.pkl"
            if creditcard_processor_path.exists():
                with open(creditcard_processor_path, 'rb') as f:
                    creditcard_processor = cloudpickle.load(f)
                print(f"Loaded Credit Card Fraud processor from {creditcard_processor_path}")
            else:
                print(f"Warning: Credit Card Fraud processor not found at {creditcard_processor_path}. Will force full preprocessing for Credit Card data.")
                creditcard_processor = None # Force full preprocessing
                creditcard_processed_df = pd.DataFrame() # Clear to force reprocessing

            if not fraud_processed_df.empty and fraud_processor is not None and \
               not creditcard_processed_df.empty and creditcard_processor is not None:
                preprocessing_skipped_successfully = True
                print("Successfully skipped preprocessing using existing files and loaded processors.")
            else:
                print("Skipping preprocessing failed for one or both datasets. Forcing full preprocessing.")

        except FileNotFoundError:
            print("Error: Processed data files not found during skip attempt. Forcing full preprocessing.")
            fraud_processed_df = pd.DataFrame() # Clear to force reprocessing
            creditcard_processed_df = pd.DataFrame() # Clear to force reprocessing
        except Exception as e:
            print(f"Error loading processed data or processors during skip attempt: {e}. Forcing full preprocessing.")
            fraud_processed_df = pd.DataFrame() # Clear to force reprocessing
            creditcard_processed_df = pd.DataFrame() # Clear to force reprocessing

    if not preprocessing_skipped_successfully: # Perform full preprocessing if not skipped successfully
        print("\n--- Starting Full Data Preprocessing ---")
        # E-commerce Fraud Data Preprocessing
        print("\nPreprocessing E-commerce Fraud Data...")
        fraud_df_deduplicated = fraud_raw_df.drop_duplicates().copy()
        X_fraud_raw = fraud_df_deduplicated.drop(columns=[FRAUD_TARGET_COL]).copy()
        y_fraud_raw = fraud_df_deduplicated[FRAUD_TARGET_COL].copy().astype(int)

        X_fraud_pre_process = X_fraud_raw.rename(columns={
            'user_id': 'CustomerId',
            'purchase_value': 'Amount',
            'purchase_time': 'TransactionStartTime'
        }).copy()
        X_fraud_merged = merge_ip_to_country(X_fraud_pre_process, ip_country_df.copy())
        if 'user_id' in fraud_df_deduplicated.columns:
            X_fraud_merged['TransactionId'] = fraud_df_deduplicated['user_id']

        fraud_processor = FraudDataProcessor(
            numerical_cols_after_rename=['Amount', 'age'], # Use the renamed column names
            categorical_cols_after_merge=['source', 'browser', 'sex', 'country'],
            time_col_after_rename='TransactionStartTime',
            signup_time_col_after_rename='signup_time',
            amount_col_after_rename='Amount',
            id_cols_for_agg_after_rename=['CustomerId', 'device_id', 'ip_address']
        )
        fraud_processed_df = fraud_processor.fit_transform(X_fraud_merged, y_fraud_raw)
        fraud_processed_df[FRAUD_TARGET_COL] = y_fraud_raw.values # Add target back for saving
        fraud_processed_df.to_csv(PROCESSED_DATA_DIR / "fraud_processed.csv", index=False)
        print(f"E-commerce Fraud Data Preprocessing Complete. Saved to {PROCESSED_DATA_DIR / 'fraud_processed.csv'}")

        # Credit Card Fraud Data Preprocessing
        print("\nPreprocessing Credit Card Fraud Data...")
        creditcard_df_deduplicated = creditcard_raw_df.drop_duplicates().copy()
        X_creditcard_raw = creditcard_df_deduplicated.drop(columns=[CREDITCARD_TARGET_COL]).copy()
        y_creditcard_raw = creditcard_df_deduplicated[CREDITCARD_TARGET_COL].copy().astype(int)

        creditcard_processor = CreditCardDataProcessor(
            numerical_cols=CREDITCARD_NUMERICAL_FEATURES
        )
        creditcard_processed_df = creditcard_processor.fit_transform(X_creditcard_raw, y_creditcard_raw)
        creditcard_processed_df[CREDITCARD_TARGET_COL] = y_creditcard_raw.values # Add target back for saving
        creditcard_processed_df.to_csv(PROCESSED_DATA_DIR / "creditcard_processed.csv", index=False)
        print(f"Credit Card Fraud Data Preprocessing Complete. Saved to {PROCESSED_DATA_DIR / 'creditcard_processed.csv'}")


    if fraud_processed_df.empty or creditcard_processed_df.empty:
        print("Processed data is empty. Exiting training pipeline.")
        return

    # --- 3. Prepare Data for Modeling ---
    # E-commerce Fraud
    X_fraud = fraud_processed_df.drop(columns=[FRAUD_TARGET_COL], errors='ignore')
    y_fraud = fraud_processed_df[FRAUD_TARGET_COL].astype(int)
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
        X_fraud, y_fraud, test_size=0.2, random_state=RANDOM_STATE, stratify=y_fraud
    )
    smote = SMOTE(random_state=RANDOM_STATE)
    print(f"Shapes before SMOTE (E-commerce): X_train_fraud: {X_train_fraud.shape}, y_train_fraud: {y_train_fraud.shape}") # Debugging print
    X_train_fraud_resampled, y_train_fraud_resampled = smote.fit_resample(X_train_fraud, y_train_fraud)

    # Credit Card Fraud
    X_creditcard = creditcard_processed_df.drop(columns=[CREDITCARD_TARGET_COL], errors='ignore')
    y_creditcard = creditcard_processed_df[CREDITCARD_TARGET_COL].astype(int)
    X_train_creditcard, X_test_creditcard, y_train_creditcard, y_test_creditcard = train_test_split(
        X_creditcard, y_creditcard, test_size=0.2, random_state=RANDOM_STATE, stratify=y_creditcard
    )
    smote = SMOTE(random_state=RANDOM_STATE)
    print(f"Shapes before SMOTE (Credit Card): X_train_creditcard: {X_train_creditcard.shape}, y_train_creditcard: {y_train_creditcard.shape}") # Debugging print
    X_train_creditcard_resampled, y_train_creditcard_resampled = smote.fit_resample(X_train_creditcard, y_train_creditcard)


    # --- 4. Model Training and MLflow Tracking ---
    best_fraud_model_estimator = None
    best_fraud_metrics = {"ROC-AUC": -1}
    best_fraud_model_name = ""

    best_creditcard_model_estimator = None
    best_creditcard_metrics = {"ROC-AUC": -1}
    best_creditcard_model_name = ""
    
    # MLflow Parent Run for E-commerce Fraud
    with mlflow.start_run(run_name="E-commerce_Fraud_Detection_Experiment") as parent_run_fraud:
        mlflow.log_param("dataset", "E-commerce Fraud")
        
        # Train Logistic Regression
        lr_strategy_fraud = LogisticRegressionStrategy(random_state=RANDOM_STATE, solver='liblinear', max_iter=1000)
        lr_metrics_fraud, lr_model_fraud = train_and_log_model(
            "Logistic Regression", X_train_fraud_resampled, y_train_fraud_resampled,
            X_test_fraud, y_test_fraud, fraud_processor, lr_strategy_fraud, "FraudDetection"
        )
        if lr_metrics_fraud["ROC-AUC"] > best_fraud_metrics["ROC-AUC"]:
            best_fraud_metrics = lr_metrics_fraud
            best_fraud_model_estimator = lr_model_fraud
            best_fraud_model_name = "Logistic Regression"
        
        # Train XGBoost Classifier
        xgb_strategy_fraud = XGBoostStrategy(
            model_type='classifier', random_state=RANDOM_STATE, n_estimators=200, learning_rate=0.1, max_depth=5
        )
        xgb_metrics_fraud, xgb_model_fraud = train_and_log_model(
            "XGBoost Classifier", X_train_fraud_resampled, y_train_fraud_resampled,
            X_test_fraud, y_test_fraud, fraud_processor, xgb_strategy_fraud, "FraudDetection"
        )
        if xgb_metrics_fraud["ROC-AUC"] > best_fraud_metrics["ROC-AUC"]:
            best_fraud_metrics = xgb_metrics_fraud
            best_fraud_model_estimator = xgb_model_fraud
            best_fraud_model_name = "XGBoost Classifier"

        # Train Random Forest Classifier
        rf_strategy_fraud = RandomForestStrategy(model_type='classifier', random_state=RANDOM_STATE, n_estimators=100, max_depth=10) # FIX: Added model_type='classifier'
        rf_metrics_fraud, rf_model_fraud = train_and_log_model(
            "Random Forest Classifier", X_train_fraud_resampled, y_train_fraud_resampled,
            X_test_fraud, y_test_fraud, fraud_processor, rf_strategy_fraud, "FraudDetection"
        )
        if rf_metrics_fraud["ROC-AUC"] > best_fraud_metrics["ROC-AUC"]:
            best_fraud_metrics = rf_metrics_fraud
            best_fraud_model_estimator = rf_model_fraud
            best_fraud_model_name = "Random Forest Classifier"

        print(f"\n--- Best Model for E-commerce Fraud: {best_fraud_model_name} with ROC-AUC: {best_fraud_metrics['ROC-AUC']:.4f} ---")
        
        # Export the best model and processor for E-commerce Fraud (local files)
        if best_fraud_model_estimator:
            model_export_path = EXPORT_DIR / "best_ecommerce_model.pkl" # Consistent name
            with open(model_export_path, 'wb') as f:
                cloudpickle.dump(best_fraud_model_estimator, f)
            print(f"Best E-commerce Fraud model exported to {model_export_path}")
            
            processor_export_path = EXPORT_DIR / "best_ecommerce_processor.pkl" # Consistent name
            if fraud_processor: # Ensure processor exists before trying to dump
                with open(processor_export_path, 'wb') as f:
                    cloudpickle.dump(fraud_processor, f)
                print(f"E-commerce Fraud processor exported to {processor_export_path}")
            else:
                print("Warning: E-commerce Fraud processor was not initialized, skipping local export.")

            # Log the locally exported processor to MLflow as an artifact
            mlflow.log_artifact(str(processor_export_path), artifact_path="processor")


    # MLflow Parent Run for Credit Card Fraud
    with mlflow.start_run(run_name="CreditCard_Fraud_Detection_Experiment") as parent_run_creditcard:
        mlflow.log_param("dataset", "Credit Card Fraud")

        # Train Logistic Regression
        lr_strategy_creditcard = LogisticRegressionStrategy(random_state=RANDOM_STATE, solver='liblinear', max_iter=1000)
        lr_metrics_creditcard, lr_model_creditcard = train_and_log_model(
            "Logistic Regression", X_train_creditcard_resampled, y_train_creditcard_resampled,
            X_test_creditcard, y_test_creditcard, creditcard_processor, lr_strategy_creditcard, "CreditCardFraud"
        )
        if lr_metrics_creditcard["ROC-AUC"] > best_creditcard_metrics["ROC-AUC"]:
            best_creditcard_metrics = lr_metrics_creditcard
            best_creditcard_model_estimator = lr_model_creditcard
            best_creditcard_model_name = "Logistic Regression"

        # Train XGBoost Classifier
        xgb_strategy_creditcard = XGBoostStrategy(
            model_type='classifier', random_state=RANDOM_STATE, n_estimators=200, learning_rate=0.1, max_depth=5
        )
        xgb_metrics_creditcard, xgb_model_creditcard = train_and_log_model(
            "XGBoost Classifier", X_train_creditcard_resampled, y_train_creditcard_resampled,
            X_test_creditcard, y_test_creditcard, creditcard_processor, xgb_strategy_creditcard, "CreditCardFraud"
        )
        if xgb_metrics_creditcard["ROC-AUC"] > best_creditcard_metrics["ROC-AUC"]:
            best_creditcard_metrics = xgb_metrics_creditcard
            best_creditcard_model_estimator = xgb_model_creditcard
            best_creditcard_model_name = "XGBoost Classifier"
            
        # Train Random Forest Classifier
        rf_strategy_creditcard = RandomForestStrategy(model_type='classifier', random_state=RANDOM_STATE, n_estimators=100, max_depth=10) # FIX: Added model_type='classifier'
        rf_metrics_creditcard, rf_model_creditcard = train_and_log_model(
            "Random Forest Classifier", X_train_creditcard_resampled, y_train_creditcard_resampled,
            X_test_creditcard, y_test_creditcard, creditcard_processor, rf_strategy_creditcard, "CreditCardFraud"
        )
        if rf_metrics_creditcard["ROC-AUC"] > best_creditcard_metrics["ROC-AUC"]:
            best_creditcard_metrics = rf_metrics_creditcard
            best_creditcard_model_estimator = rf_model_creditcard
            best_creditcard_model_name = "Random Forest Classifier"

        print(f"\n--- Best Model for Credit Card Fraud: {best_creditcard_model_name} with ROC-AUC: {best_creditcard_metrics['ROC-AUC']:.4f} ---")

        if best_creditcard_model_estimator:
            model_export_path = EXPORT_DIR / "best_creditcard_model.pkl" # Consistent name
            with open(model_export_path, 'wb') as f:
                cloudpickle.dump(best_creditcard_model_estimator, f)
            print(f"Best Credit Card Fraud model exported to {model_export_path}")

            processor_export_path = EXPORT_DIR / "best_creditcard_processor.pkl" # Consistent name
            if creditcard_processor: # Ensure processor exists before trying to dump
                with open(processor_export_path, 'wb') as f:
                    cloudpickle.dump(creditcard_processor, f)
                print(f"Credit Card Fraud processor exported to {processor_export_path}")
            else:
                print("Warning: Credit Card Fraud processor was not initialized, skipping local export.")
            
            # Log the locally exported processor to MLflow as an artifact
            mlflow.log_artifact(str(processor_export_path), artifact_path="processor")

    print("\nComprehensive model training and tracking process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fraud detection model training pipeline.")
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip reprocessing raw data if preprocessed files and processor are found in data/processed.')
    args = parser.parse_args()

    # Define paths for raw data
    FRAUD_RAW_DATA_PATH = project_root / "data" / "raw" / "Fraud_Data.csv"
    IP_TO_COUNTRY_RAW_DATA_PATH = project_root / "data" / "raw" / "IpAddress_to_Country.csv"
    CREDITCARD_RAW_DATA_PATH = project_root / "data" / "raw" / "creditcard.csv"

    run_training_pipeline(
        fraud_raw_data_path=FRAUD_RAW_DATA_PATH,
        ip_to_country_raw_data_path=IP_TO_COUNTRY_RAW_DATA_PATH,
        creditcard_raw_data_path=CREDITCARD_RAW_DATA_PATH,
        skip_preprocessing=args.skip_preprocessing
    )
