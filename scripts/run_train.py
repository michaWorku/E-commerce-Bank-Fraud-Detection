import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from pathlib import Path
import sys
import json
from tqdm import tqdm 
import os 
import cloudpickle

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data
from src.data_processing import DataProcessor
from src.models.model_trainer import ModelTrainer
from src.models.logistic_regression_strategy import LogisticRegressionStrategy
from src.models.decision_tree_strategy import DecisionTreeStrategy
from src.models.random_forest_strategy import RandomForestStrategy
from src.models.xgboost_strategy import XGBoostStrategy
from src.models.model_evaluator import evaluate_classification_model

# Define paths
RAW_DATA_PATH = project_root / "data" / "raw" / "data.csv"
MLRUNS_PATH = project_root / "mlruns" # MLflow tracking directory
EXPORT_DIR = project_root / "exported_model" # New directory for exported model and processor

# --- Ensure MLflow tracking directory exists (for local file tracking) ---
MLRUNS_PATH.mkdir(parents=True, exist_ok=True) 

# --- Ensure export directory exists ---
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Set MLflow tracking URI to use local file system for experiment tracking
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")
mlflow.set_experiment("Credit Risk Probability Model Training")


def train_and_evaluate_model(
    model_strategy,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: dict,
    model_name: str
):
    """
    Trains, tunes, evaluates a model, and logs results to MLflow.

    Args:
        model_strategy: An instance of BaseModelStrategy (e.g., LogisticRegressionStrategy).
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        model_name (str): Name of the model for MLflow logging.
    """
    with mlflow.start_run(run_name=f"{model_name}_GridSearch", nested=True) as run: # Use nested=True and capture run
        mlflow.log_param("model_type", model_name)

        trainer = ModelTrainer(model_strategy)

        print(f"\n--- Starting Hyperparameter Tuning for {model_name} ---")
        estimator_for_grid = model_strategy.get_model()

        grid_search = GridSearchCV(
            estimator=estimator_for_grid,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimator = grid_search.best_estimator_

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best cross-validation ROC-AUC for {model_name}: {best_score:.4f}")

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_roc_auc", best_score)

        if isinstance(model_strategy, LogisticRegressionStrategy):
            trainer.set_strategy(LogisticRegressionStrategy(random_state=42, **best_params))
        elif isinstance(model_strategy, DecisionTreeStrategy):
            trainer.set_strategy(DecisionTreeStrategy(model_type='classifier', random_state=42, **best_params))
        elif isinstance(model_strategy, RandomForestStrategy):
            trainer.set_strategy(RandomForestStrategy(model_type='classifier', random_state=42, **best_params))
        elif isinstance(model_strategy, XGBoostStrategy):
            trainer.set_strategy(XGBoostStrategy(model_type='classifier', random_state=42, **best_params))
        
        trainer.train_model(X_train, y_train)
        
        y_pred_proba = trainer.predict_model(X_test)

        print(f"\n--- Evaluating {model_name} on Test Set ---")
        metrics = evaluate_classification_model(y_test.values, y_pred_proba)

        mlflow.log_metrics(metrics)

        # Log model artifacts to the current run's artifact URI
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            name="model", # Use 'name' for logging within the run's artifacts
            signature=mlflow.models.infer_signature(X_test, best_estimator.predict_proba(X_test))
        )
        print(f"Model '{model_name}' logged to MLflow run.")
        
        return best_score, best_estimator, run.info.run_id # Return run_id

def main():
    print("Starting model training and tracking process...")

    # --- 1. Load and Process Data ---
    print("Loading raw data...")
    df_raw = load_data(RAW_DATA_PATH, delimiter=',')
    if df_raw is None or df_raw.empty:
        print("Error: Raw data could not be loaded or is empty. Exiting.")
        return

    if 'FraudResult' not in df_raw.columns:
        print("Error: 'FraudResult' column not found in raw data. Cannot proceed with training. Exiting.")
        return
    
    X_raw = df_raw.drop(columns=['FraudResult'])
    y_raw = df_raw['FraudResult']

    id_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProductId', 'CurrencyCode', 'CountryCode']
    numerical_features = ['Amount', 'Value', 'PricingStrategy']
    categorical_features = ['ProviderId', 'ProductCategory', 'ChannelId']
    time_column = 'TransactionStartTime'

    processor = DataProcessor(
        numerical_cols=numerical_features,
        categorical_cols=categorical_features,
        time_column=time_column,
        id_columns=id_columns,
        target_column='FraudResult'
    )

    print("Processing data...")
    X_processed = processor.fit_transform(X_raw.copy(), y_raw.copy())
    print("Data processing complete.")

    # --- Export the DataProcessor for API deployment ---
    processor_export_path = EXPORT_DIR / "data_processor.pkl"
    with open(processor_export_path, "wb") as f:
        cloudpickle.dump(processor, f)
    print(f"DataProcessor exported to {processor_export_path}")

    # --- Log the DataProcessor as an artifact for MLflow tracking (for run_predict/interpret) ---
    with mlflow.start_run(run_name="Log_DataProcessor_Artifact", nested=True):
        processor_artifact_path = "data_processor.pkl" # Temp file for logging
        with open(processor_artifact_path, "wb") as f:
            cloudpickle.dump(processor, f)
        mlflow.log_artifact(processor_artifact_path, artifact_path="processor") # Log as 'processor' artifact
        os.remove(processor_artifact_path) # Clean up local file
        print("DataProcessor logged as MLflow artifact.")


    if 'is_high_risk' not in X_processed.columns:
        print("Error: 'is_high_risk' column not found after data processing. Cannot proceed with training. Exiting.")
        return
    
    feature_cols = [col for col in X_processed.columns if col not in id_columns and col != 'is_high_risk' and col != 'FraudResult']
    
    X_final = X_processed[feature_cols]
    y_final = X_processed['is_high_risk']

    non_numeric_cols = X_final.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found in final features: {non_numeric_cols}. Dropping them.")
        X_final = X_final.drop(columns=non_numeric_cols)

    if X_final.isnull().any().any():
        print("Warning: NaNs detected in final feature set. Imputing with median.")
        for col in X_final.columns:
            if X_final[col].isnull().any():
                if not X_final[col].isnull().all():
                    X_final[col] = X_final[col].fillna(X_final[col].median())
                else:
                    X_final[col] = X_final[col].fillna(0.0)
    
    y_final = pd.to_numeric(y_final, errors='coerce').astype(int)
    if y_final.isnull().any():
        print("Error: NaNs detected in target variable after processing. Cannot proceed. Exiting.")
        return
    if y_final.nunique() < 2:
        print("Error: Target variable has less than two unique classes. Cannot perform classification. Exiting.")
        return


    # --- 2. Split Data ---
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )
    print(f"Train set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


    # --- 3. Model Selection and Training with Hyperparameter Tuning and MLflow Tracking ---
    best_model_overall_score = -1
    best_model_overall = None
    best_model_overall_name = ""
    best_model_run_id = None # Store the run_id of the best model

    models_to_train = [
        ("Logistic Regression Classifier", LogisticRegressionStrategy(random_state=42), {
            'solver': ['liblinear', 'lbfgs'], 'C': [0.1, 1.0, 10.0], 'max_iter': [100, 200]
        }),
        ("Decision Tree Classifier", DecisionTreeStrategy(model_type='classifier', random_state=42), {
            'max_depth': [5, 10, 15], 'min_samples_leaf': [1, 5, 10]
        }),
        ("Random Forest Classifier", RandomForestStrategy(model_type='classifier', random_state=42), {
            'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_leaf': [1, 5]
        }),
        ("XGBoost Classifier", XGBoostStrategy(model_type='classifier', random_state=42), {
            'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]
        })
    ]

    print("\n--- Training and Evaluating Models ---")
    for model_name, strategy_instance, param_grid in tqdm(models_to_train, desc="Overall Model Training Progress"):
        current_best_score, current_best_estimator, current_run_id = train_and_evaluate_model(
            strategy_instance, X_train, y_train, X_test, y_test, param_grid, model_name
        )
        if current_best_score > best_model_overall_score:
            best_model_overall_score = current_best_score
            best_model_overall = current_best_estimator
            best_model_overall_name = model_name
            best_model_run_id = current_run_id # Store the run_id

    print(f"\n--- Best Model Overall: {best_model_overall_name} with ROC-AUC: {best_model_overall_score:.4f} ---")

    # --- 4. Register Best Model in MLflow Model Registry (for run_predict/interpret) ---
    if best_model_overall is not None:
        with mlflow.start_run(run_name="Register_Best_Model_Final", nested=True) as final_run:
            # Register the model using its run_id and artifact_path
            # This is crucial for run_predict.py and run_interpret.py
            mlflow.register_model(
                model_uri=f"runs:/{best_model_run_id}/model", # Point to the artifact logged in its run
                name="CreditRiskClassifier"
            )
            mlflow.log_param("final_best_model_name", best_model_overall_name)
            mlflow.log_metric("final_best_roc_auc", best_model_overall_score)
            print(f"Best model '{best_model_overall_name}' registered in MLflow Model Registry as 'CreditRiskClassifier'.")
    else:
        print("No best model identified for registration.")

    # --- Export the Best Model for API deployment ---
    if best_model_overall is not None:
        model_export_path = EXPORT_DIR / "best_model.pkl"
        with open(model_export_path, "wb") as f:
            cloudpickle.dump(best_model_overall, f)
        print(f"Best model '{best_model_overall_name}' exported to {model_export_path}")
    else:
        print("No best model identified for export.")

    print("\nModel training and tracking process completed.")


if __name__ == "__main__":
    main()
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

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_processing.loader import load_data
from src.data_processing.preprocessor import FraudDataProcessor, CreditCardDataProcessor # Import specific processors
from src.models.model_trainer import ModelTrainer
from src.models.logistic_regression_strategy import LogisticRegressionStrategy
from src.models.decision_tree_strategy import DecisionTreeStrategy # Keep if needed for other experiments
from src.models.random_forest_strategy import RandomForestStrategy # Keep if needed for other experiments
from src.models.xgboost_strategy import XGBoostStrategy
from src.models.model_evaluator import evaluate_classification_model

# Define paths
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
MLRUNS_PATH = project_root / "mlruns" # MLflow tracking directory
EXPORT_DIR = project_root / "exported_models" # New directory for exported models and processors

# --- Ensure directories exist ---
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_PATH.mkdir(parents=True, exist_ok=True) 
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Set MLflow tracking URI to use local file system for experiment tracking
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")
mlflow.set_experiment("Fraud Detection Model Training")

# Global Random State for reproducibility
RANDOM_STATE = 42

def process_and_split_data(
    raw_df: pd.DataFrame, 
    target_col: str, 
    dataset_name: str,
    processor_class, # Pass the specific processor class (e.g., FraudDataProcessor)
    processor_params: dict, # Parameters for the processor
    test_size: float = 0.2, 
    random_state: int = RANDOM_STATE
):
    """
    Loads raw data, processes it using the specified processor,
    and performs a stratified train-test split with SMOTE on training data.

    Args:
        raw_df (pd.DataFrame): The raw DataFrame to process.
        target_col (str): The name of the target column.
        dataset_name (str): Name of the dataset (e.g., 'E-commerce Fraud', 'Credit Card Fraud').
        processor_class: The DataProcessor class to instantiate (e.g., FraudDataProcessor).
        processor_params (dict): Dictionary of parameters to initialize the processor.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train_resampled, X_test, y_train_resampled, y_test, fitted_processor
               Returns empty DataFrames/Series and None if processing fails.
    """
    print(f"\n--- Data Preparation: {dataset_name} ---")

    if raw_df.empty:
        print(f"Error: Raw data for {dataset_name} is empty. Skipping data preparation.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), None

    if target_col not in raw_df.columns:
        print(f"Error: Target column '{target_col}' not found in {dataset_name} raw data. Exiting.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), None

    X_raw = raw_df.drop(columns=[target_col]).copy()
    y_raw = raw_df[target_col].copy()

    # Ensure target is integer type for stratification
    y_raw = y_raw.astype(int)

    print(f"Original {dataset_name} Data shape: {X_raw.shape}")
    print(f"Original {dataset_name} Target distribution:\n{y_raw.value_counts(normalize=True)}")

    processor = processor_class(**processor_params)
    print(f"Applying preprocessing pipeline for {dataset_name}...")
    
    # Fit and transform the data
    X_processed = processor.fit_transform(X_raw, y_raw)
    
    # After preprocessing, ensure all features are numeric and handle any remaining NaNs
    # This is a safeguard, as preprocessor should handle most.
    non_numeric_cols = X_processed.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found after processing for {dataset_name}: {non_numeric_cols}. Attempting to convert/drop.")
        for col in non_numeric_cols:
            try:
                # Try to convert to numeric, coerce errors, then fillna
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0.0)
            except Exception as e:
                print(f"Could not convert column {col} to numeric: {e}. Dropping column.")
                X_processed = X_processed.drop(columns=[col])

    if X_processed.isnull().any().any():
        print(f"Warning: NaNs detected in processed features for {dataset_name}. Imputing with median/0.")
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                if X_processed[col].dtype == 'object': # If it's still object, fill with mode or drop
                    X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0] if not X_processed[col].mode().empty else 'missing')
                elif pd.api.types.is_numeric_dtype(X_processed[col]):
                    X_processed[col] = X_processed[col].fillna(X_processed[col].median() if not X_processed[col].empty else 0.0)
                else: # For other types, just fill with a placeholder
                    X_processed[col] = X_processed[col].fillna(0) # Generic fill for other types

    print(f"Processed {dataset_name} Data shape: {X_processed.shape}")
    print(f"Processed {dataset_name} Data dtypes:\n{X_processed.dtypes}")

    # Train-Test Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_raw, test_size=test_size, random_state=random_state, stratify=y_raw
    )

    print(f"\n{dataset_name} Train set shape: {X_train.shape}")
    print(f"{dataset_name} Test set shape: {X_test.shape}")
    print(f"{dataset_name} Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"{dataset_name} Test target distribution:\n{y_test.value_counts(normalize=True)}")

    # Apply SMOTE to the training data only
    print(f"\nApplying SMOTE to {dataset_name} training data...")
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"{dataset_name} Train set shape after SMOTE: {X_train_resampled.shape}")
    print(f"{dataset_name} Train target distribution after SMOTE:\n{y_train_resampled.value_counts(normalize=True)}")
    
    return X_train_resampled, X_test, y_train_resampled, y_test, processor


def train_and_evaluate_model(
    model_strategy,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: dict,
    model_name: str,
    dataset_tag: str # e.g., 'E-commerce', 'CreditCard'
):
    """
    Trains, tunes, evaluates a model, and logs results to MLflow.

    Args:
        model_strategy: An instance of BaseModelStrategy (e.g., LogisticRegressionStrategy).
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        model_name (str): Name of the model for MLflow logging.
        dataset_tag (str): Tag to identify the dataset (e.g., 'E-commerce', 'CreditCard').

    Returns:
        tuple: best_score, best_estimator, run_id
               Returns None, None, None if training fails.
    """
    with mlflow.start_run(run_name=f"{dataset_tag}_{model_name}_Training", nested=True) as run:
        mlflow.log_param("dataset", dataset_tag)
        mlflow.log_param("model_type", model_name)

        trainer = ModelTrainer(model_strategy)

        print(f"\n--- Starting Hyperparameter Tuning for {model_name} on {dataset_tag} Data ---")
        estimator_for_grid = model_strategy.get_model()

        try:
            grid_search = GridSearchCV(
                estimator=estimator_for_grid,
                param_grid=param_grid,
                cv=3, # Using 3-fold cross-validation for speed
                scoring='roc_auc', # Optimize for ROC-AUC during tuning
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            best_estimator = grid_search.best_estimator_

            print(f"Best parameters for {model_name} ({dataset_tag}): {best_params}")
            print(f"Best cross-validation ROC-AUC for {model_name} ({dataset_tag}): {best_score:.4f}")

            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_roc_auc", best_score)

            # Re-initialize trainer with best parameters and train on full resampled training data
            if isinstance(model_strategy, LogisticRegressionStrategy):
                trainer.set_strategy(LogisticRegressionStrategy(random_state=RANDOM_STATE, **best_params))
            elif isinstance(model_strategy, XGBoostStrategy):
                trainer.set_strategy(XGBoostStrategy(model_type='classifier', random_state=RANDOM_STATE, **best_params))
            # Add other strategies if you include them (e.g., DecisionTreeStrategy, RandomForestStrategy)
            elif isinstance(model_strategy, DecisionTreeStrategy):
                trainer.set_strategy(DecisionTreeStrategy(model_type='classifier', random_state=RANDOM_STATE, **best_params))
            elif isinstance(model_strategy, RandomForestStrategy):
                trainer.set_strategy(RandomForestStrategy(model_type='classifier', random_state=RANDOM_STATE, **best_params))
            
            trainer.train_model(X_train, y_train)
            
            y_pred_proba = trainer.predict_model(X_test)

            print(f"\n--- Evaluating {model_name} on {dataset_tag} Test Set ---")
            metrics = evaluate_classification_model(y_test.values, y_pred_proba)

            mlflow.log_metrics(metrics)

            # Log model artifacts to the current run's artifact URI
            mlflow.sklearn.log_model(
                sk_model=best_estimator,
                artifact_path="model", # Use 'model' as artifact path within the run
                signature=mlflow.models.infer_signature(X_test, best_estimator.predict_proba(X_test)),
                input_example=X_test.head(1) # Log an input example
            )
            print(f"Model '{model_name}' for {dataset_tag} logged to MLflow run.")
            
            return metrics.get('ROC-AUC', -1), best_estimator, run.info.run_id # Return ROC-AUC for best model selection

        except Exception as e:
            print(f"Error during training and evaluation of {model_name} on {dataset_tag} data: {e}")
            mlflow.log_param("training_status", "failed")
            mlflow.log_param("error_message", str(e))
            return -1, None, None


def main():
    print("Starting comprehensive model training and tracking process...")

    # --- E-commerce Fraud Data (Fraud_Data.csv) ---
    fraud_data_path = RAW_DATA_DIR / "Fraud_Data.csv"
    ip_to_country_path = RAW_DATA_DIR / "IpAddress_to_Country.csv"

    # Load raw data for E-commerce Fraud
    fraud_df_raw = load_data(fraud_data_path, use_gpu=False)
    ip_country_df_raw = load_data(ip_to_country_path, use_gpu=False)

    if fraud_df_raw.empty or ip_country_df_raw.empty:
        print("Skipping E-commerce Fraud data processing due to missing raw data.")
        X_train_fraud_resampled, X_test_fraud, y_train_fraud_resampled, y_test_fraud, fraud_processor = \
            pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), None
    else:
        # Simulate renaming and IP merge for FraudDataProcessor
        # This part should ideally be done by a separate initial transform if not already done in run_data_pipeline.py
        # For simplicity, we'll do it here before passing to the processor.
        fraud_df_merged = fraud_df_raw.rename(columns={
            'user_id': 'CustomerId',
            'purchase_value': 'Amount',
            'purchase_time': 'TransactionStartTime',
            'user_id': 'TransactionId' # Using user_id as a proxy for TransactionId for frequency count
        }).copy()
        from src.utils.helpers import merge_ip_to_country # Ensure this is imported
        fraud_df_merged = merge_ip_to_country(fraud_df_merged, ip_country_df_raw.copy())
        fraud_df_merged['class'] = fraud_df_raw['class'] # Re-attach target for processing

        fraud_processor_params = {
            'numerical_cols_after_rename': ['Amount', 'age'],
            'categorical_cols_after_merge': ['source', 'browser', 'sex', 'country'],
            'time_col_after_rename': 'TransactionStartTime',
            'signup_time_col_after_rename': 'signup_time',
            'amount_col_after_rename': 'Amount',
            'id_cols_for_agg_after_rename': ['CustomerId', 'device_id', 'ip_address']
        }
        X_train_fraud_resampled, X_test_fraud, y_train_fraud_resampled, y_test_fraud, fraud_processor = \
            process_and_split_data(
                fraud_df_merged, 'class', 'E-commerce Fraud',
                FraudDataProcessor, fraud_processor_params
            )

    # --- Bank Credit Card Fraud Data (creditcard.csv) ---
    creditcard_data_path = RAW_DATA_DIR / "creditcard.csv"

    # Load raw data for Credit Card Fraud
    creditcard_df_raw = load_data(creditcard_data_path, use_gpu=False)

    if creditcard_df_raw.empty:
        print("Skipping Credit Card Fraud data processing due to missing raw data.")
        X_train_creditcard_resampled, X_test_creditcard, y_train_creditcard_resampled, y_test_creditcard, creditcard_processor = \
            pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), None
    else:
        # For CreditCardDataProcessor, the raw data columns are directly used as numerical features
        creditcard_numerical_features = [col for col in creditcard_df_raw.columns if col not in ['Time', 'Amount', 'Class']] + ['Time', 'Amount']
        creditcard_processor_params = {
            'numerical_cols': creditcard_numerical_features
        }
        X_train_creditcard_resampled, X_test_creditcard, y_train_creditcard_resampled, y_test_creditcard, creditcard_processor = \
            process_and_split_data(
                creditcard_df_raw, 'Class', 'Credit Card Fraud',
                CreditCardDataProcessor, creditcard_processor_params
            )

    # --- Models to Train ---
    models_config = {
        "Logistic Regression": {
            "strategy": LogisticRegressionStrategy(random_state=RANDOM_STATE),
            "param_grid": {'solver': ['liblinear'], 'C': [0.1, 1.0, 10.0]} # Reduced grid for faster demo
        },
        "XGBoost Classifier": {
            "strategy": XGBoostStrategy(model_type='classifier', random_state=RANDOM_STATE),
            "param_grid": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]} # Reduced grid
        }
    }

    # --- Train and Evaluate Models for E-commerce Fraud ---
    best_fraud_model_score = -1
    best_fraud_model_estimator = None
    best_fraud_model_name = ""
    best_fraud_model_run_id = None
    best_fraud_processor = None

    if not X_train_fraud_resampled.empty:
        print("\n--- Training Models for E-commerce Fraud Data ---")
        for model_name, config in tqdm(models_config.items(), desc="E-commerce Models"):
            score, estimator, run_id = train_and_evaluate_model(
                config["strategy"], X_train_fraud_resampled, y_train_fraud_resampled, 
                X_test_fraud, y_test_fraud, config["param_grid"], model_name, 'E-commerce'
            )
            if score > best_fraud_model_score:
                best_fraud_model_score = score
                best_fraud_model_estimator = estimator
                best_fraud_model_name = model_name
                best_fraud_model_run_id = run_id
                best_fraud_processor = fraud_processor # Store the processor used for the best model

        print(f"\n--- Best Model for E-commerce Fraud: {best_fraud_model_name} with ROC-AUC: {best_fraud_model_score:.4f} ---")

        # Register Best E-commerce Fraud Model and its Processor
        if best_fraud_model_estimator is not None:
            with mlflow.start_run(run_name="Register_Best_E-commerce_Fraud_Model", nested=True) as final_run:
                mlflow.register_model(
                    model_uri=f"runs:/{best_fraud_model_run_id}/model",
                    name="FraudDetection_XGBoost" if "XGBoost" in best_fraud_model_name else "FraudDetection_LogisticRegression"
                )
                mlflow.log_param("final_best_model_name", best_fraud_model_name)
                mlflow.log_metric("final_best_roc_auc", best_fraud_model_score)
                print(f"Best E-commerce Fraud model '{best_fraud_model_name}' registered in MLflow Model Registry.")

                # Log the processor used for this dataset as an artifact under the final run
                processor_artifact_path = EXPORT_DIR / f"fraud_processor_{final_run.info.run_id}.pkl"
                with open(processor_artifact_path, "wb") as f:
                    cloudpickle.dump(best_fraud_processor, f)
                mlflow.log_artifact(str(processor_artifact_path), artifact_path="processor")
                os.remove(processor_artifact_path) # Clean up local file
                print(f"FraudDataProcessor logged as MLflow artifact for E-commerce Fraud.")

                # Export the best model for E-commerce Fraud
                model_export_path = EXPORT_DIR / "best_fraud_model.pkl"
                with open(model_export_path, "wb") as f:
                    cloudpickle.dump(best_fraud_model_estimator, f)
                print(f"Best E-commerce Fraud model exported to {model_export_path}")
        else:
            print("No best E-commerce Fraud model identified for registration/export.")
    else:
        print("Skipping E-commerce Fraud model training due to empty data.")


    # --- Train and Evaluate Models for Credit Card Fraud ---
    best_creditcard_model_score = -1
    best_creditcard_model_estimator = None
    best_creditcard_model_name = ""
    best_creditcard_model_run_id = None
    best_creditcard_processor = None

    if not X_train_creditcard_resampled.empty:
        print("\n--- Training Models for Credit Card Fraud Data ---")
        for model_name, config in tqdm(models_config.items(), desc="Credit Card Models"):
            score, estimator, run_id = train_and_evaluate_model(
                config["strategy"], X_train_creditcard_resampled, y_train_creditcard_resampled,
                X_test_creditcard, y_test_creditcard, config["param_grid"], model_name, 'CreditCard'
            )
            if score > best_creditcard_model_score:
                best_creditcard_model_score = score
                best_creditcard_model_estimator = estimator
                best_creditcard_model_name = model_name
                best_creditcard_model_run_id = run_id
                best_creditcard_processor = creditcard_processor # Store the processor used for the best model

        print(f"\n--- Best Model for Credit Card Fraud: {best_creditcard_model_name} with ROC-AUC: {best_creditcard_model_score:.4f} ---")

        # Register Best Credit Card Fraud Model and its Processor
        if best_creditcard_model_estimator is not None:
            with mlflow.start_run(run_name="Register_Best_CreditCard_Fraud_Model", nested=True) as final_run:
                mlflow.register_model(
                    model_uri=f"runs:/{best_creditcard_model_run_id}/model",
                    name="CreditCardFraud_XGBoost" if "XGBoost" in best_creditcard_model_name else "CreditCardFraud_LogisticRegression"
                )
                mlflow.log_param("final_best_model_name", best_creditcard_model_name)
                mlflow.log_metric("final_best_roc_auc", best_creditcard_model_score)
                print(f"Best Credit Card Fraud model '{best_creditcard_model_name}' registered in MLflow Model Registry.")

                # Log the processor used for this dataset as an artifact under the final run
                processor_artifact_path = EXPORT_DIR / f"creditcard_processor_{final_run.info.run_id}.pkl"
                with open(processor_artifact_path, "wb") as f:
                    cloudpickle.dump(best_creditcard_processor, f)
                mlflow.log_artifact(str(processor_artifact_path), artifact_path="processor")
                os.remove(processor_artifact_path) # Clean up local file
                print(f"CreditCardDataProcessor logged as MLflow artifact for Credit Card Fraud.")

                # Export the best model for Credit Card Fraud
                model_export_path = EXPORT_DIR / "best_creditcard_model.pkl"
                with open(model_export_path, "wb") as f:
                    cloudpickle.dump(best_creditcard_model_estimator, f)
                print(f"Best Credit Card Fraud model exported to {model_export_path}")
        else:
            print("No best Credit Card Fraud model identified for registration/export.")
    else:
        print("Skipping Credit Card Fraud model training due to empty data.")

    print("\nComprehensive model training and tracking process completed.")


if __name__ == "__main__":
    main()
