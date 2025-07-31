import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import sys
import datetime

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import from the new loader path
from src.data_processing.loader import load_data

# Import feature engineering transformers from the new module
from src.feature_engineering.engineer import HandleAmountFeatures, TemporalFeatureEngineer, TransactionFrequencyVelocity


class RemoveDuplicates(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove duplicate rows from a DataFrame.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.empty:
            print("RemoveDuplicates: Input DataFrame is empty. Returning empty DataFrame.")
            return X.copy()
        
        initial_rows = X.shape[0]
        X_deduplicated = X.drop_duplicates().copy()
        rows_removed = initial_rows - X_deduplicated.shape[0]
        if rows_removed > 0:
            print(f"Removed {rows_removed} duplicate rows.")
        else:
            print("No duplicate rows found.")
        return X_deduplicated


class DataTypeCorrector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to correct data types for specified columns.
    Specifically handles datetime conversion and ensures numerical types.
    """
    def __init__(self, datetime_cols: list = None, numerical_cols: list = None):
        self.datetime_cols = datetime_cols if datetime_cols is not None else []
        self.numerical_cols = numerical_cols if numerical_cols is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if X_copy.empty:
            print("DataTypeCorrector: Input DataFrame is empty. Returning empty DataFrame.")
            return X_copy

        # Convert specified columns to datetime
        for col in self.datetime_cols:
            if col in X_copy.columns:
                print(f"Converting '{col}' to datetime using pandas...")
                temp_series = pd.to_datetime(X_copy[col], errors='coerce', utc=True)
                # Fill NaT with a default datetime
                X_copy[col] = temp_series.fillna(pd.Timestamp('1970-01-01', tz='UTC'))
            else:
                print(f"Warning: Datetime column '{col}' not found for type correction.")

        # Convert specified columns to numerical
        for col in self.numerical_cols:
            if col in X_copy.columns:
                print(f"Converting '{col}' to numerical using pandas...")
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0.0).astype(float)
            else:
                print(f"Warning: Numerical column '{col}' not found for type correction.")
        
        return X_copy


class EnsureNumericalDtypes(BaseEstimator, TransformerMixin):
    """
    A robust transformer to ensure all columns that can be numerical are indeed numerical (float).
    Coerces errors and fills NaNs with 0.0.
    This acts as a final safeguard before scaling/encoding.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if X_copy.empty:
            print("EnsureNumericalDtypes: Input DataFrame is empty. Returning empty DataFrame.")
            return X_copy

        print("Ensuring all numerical-like columns are float and handling NaNs...")
        for col in X_copy.columns:
            # Check if column is already numeric, or if it's object type that might contain numbers
            if pd.api.types.is_numeric_dtype(X_copy[col]) or pd.api.types.is_object_dtype(X_copy[col]):
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0.0).astype(float)
            elif pd.api.types.is_datetime64_any_dtype(X_copy[col]):
                # Convert datetime to numerical (Unix timestamp or days since epoch)
                print(f"Warning: Datetime column '{col}' found in EnsureNumericalDtypes. Converting to numeric (Unix timestamp).")
                X_copy[col] = pd.Series(X_copy[col].astype('int64') // 10**9, index=X_copy.index).fillna(0.0).astype(float) # Unix timestamp in seconds
        return X_copy


class FinalFeatureProcessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies final imputation, scaling of numerical features,
    and One-Hot Encoding of categorical features.
    Ensures correct numerical dtypes after transformation.
    """
    def __init__(self, numerical_cols_to_scale: list, categorical_cols_to_encode: list,
                 imputation_strategy_num: str = 'median', imputation_strategy_cat: str = 'most_frequent'):
        self.numerical_cols_to_scale = numerical_cols_to_scale
        self.categorical_cols_to_encode = categorical_cols_to_encode
        self.imputation_strategy_num = imputation_strategy_num
        self.imputation_strategy_cat = imputation_strategy_cat
        self.preprocessor = None
        self.fitted_feature_names_out = None

    def fit(self, X, y=None):
        if X.empty:
            print("FinalFeatureProcessor: Input DataFrame is empty. Skipping fit.")
            self.fitted_feature_names_out = [] # No features if empty
            return self

        X_pd = X.copy() # Already pandas
        y_pd = y.copy() if y is not None else y

        # Identify columns that actually exist in the input X_pd
        actual_numerical_cols = [col for col in self.numerical_cols_to_scale if col in X_pd.columns]
        actual_categorical_cols = [col for col in self.categorical_cols_to_encode if col in X_pd.columns]

        transformers = []

        if actual_numerical_cols:
            transformers.append(('num_pipeline',
                                 Pipeline(steps=[
                                     ('imputer', SimpleImputer(strategy=self.imputation_strategy_num)),
                                     ('scaler', StandardScaler())
                                 ]),
                                 actual_numerical_cols))

        if actual_categorical_cols:
            transformers.append(('cat_pipeline',
                                 Pipeline(steps=[
                                     ('imputer', SimpleImputer(strategy=self.imputation_strategy_cat)),
                                     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                                 ]),
                                 actual_categorical_cols))

        if not transformers:
            self.preprocessor = 'passthrough'
            self.fitted_feature_names_out = X_pd.columns.tolist() # Keep all columns if no transformers
        else:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough', # Keep other columns not specified in transformers
                verbose_feature_names_out=False # Ensures cleaner output column names
            )
            self.preprocessor.fit(X_pd, y_pd)
            self.fitted_feature_names_out = self.preprocessor.get_feature_names_out(X_pd.columns)

        return self

    def transform(self, X):
        if X.empty:
            print("FinalFeatureProcessor: Input DataFrame is empty. Returning empty DataFrame.")
            if self.fitted_feature_names_out is not None:
                return pd.DataFrame(columns=self.fitted_feature_names_out)
            else:
                return X.copy()
        
        if self.preprocessor is None: # Case where fit was called on an empty dataframe
            print("FinalFeatureProcessor: Transformer not fitted. Returning original DataFrame.")
            return X.copy()
        
        if self.preprocessor == 'passthrough':
            return X.copy()

        X_pd = X.copy() # Already pandas

        X_transformed_array = self.preprocessor.transform(X_pd)

        if self.fitted_feature_names_out is None:
            feature_names = self.preprocessor.get_feature_names_out(X_pd.columns) if hasattr(self.preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_transformed_array.shape[1])]
        else:
            feature_names = self.fitted_feature_names_out

        X_transformed_df = pd.DataFrame(X_transformed_array, columns=feature_names, index=X.index)
        
        # Explicitly cast all numerical columns to float after transformation.
        for col in X_transformed_df.columns:
            if pd.api.types.is_numeric_dtype(X_transformed_df[col]) or pd.api.types.is_object_dtype(X_transformed_df[col]):
                X_transformed_df[col] = pd.to_numeric(X_transformed_df[col], errors='coerce').fillna(0.0).astype(float)
        
        return X_transformed_df


class FraudDataProcessor:
    """
    A specific data processor for Fraud_Data.csv, orchestrating relevant transformers.
    It expects column names *after* initial renaming (e.g., 'Amount' instead of 'purchase_value').
    """
    def __init__(self, numerical_cols_after_rename: list, categorical_cols_after_merge: list,
                 time_col_after_rename: str, signup_time_col_after_rename: str,
                 amount_col_after_rename: str, id_cols_for_agg_after_rename: list):
        
        self.numerical_cols_after_rename = numerical_cols_after_rename
        self.categorical_cols_after_merge = categorical_cols_after_merge
        self.time_col_after_rename = time_col_after_rename
        self.signup_time_col_after_rename = signup_time_col_after_rename
        self.amount_col_after_rename = amount_col_after_rename
        self.id_cols_for_agg_after_rename = id_cols_for_agg_after_rename

        self.final_numerical_features = []
        self.final_categorical_features = []

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """
        Builds the scikit-learn pipeline for Fraud_Data.csv preprocessing and feature engineering.
        """
        pipeline_steps = [
            ('remove_duplicates', RemoveDuplicates()),
            ('data_type_corrector', DataTypeCorrector(
                datetime_cols=[self.time_col_after_rename, self.signup_time_col_after_rename],
                numerical_cols=self.numerical_cols_after_rename + [self.amount_col_after_rename]
            )),
            ('handle_amount_features', HandleAmountFeatures(amount_col=self.amount_col_after_rename)),
            ('transaction_frequency_velocity', TransactionFrequencyVelocity(
                id_cols=self.id_cols_for_agg_after_rename,
                time_col=self.time_col_after_rename,
                amount_col=self.amount_col_after_rename
            )),
            ('temporal_feature_engineer', TemporalFeatureEngineer(
                purchase_time_col=self.time_col_after_rename,
                signup_time_col=self.signup_time_col_after_rename
            )),
            ('ensure_numerical_dtypes', EnsureNumericalDtypes())
        ]
        
        # Re-calculate final numerical features based on expected outputs of previous steps
        self.final_numerical_features = list(self.numerical_cols_after_rename)
        self.final_numerical_features.append('IsRefund') # From HandleAmountFeatures

        # From TemporalFeatureEngineer
        self.final_numerical_features.extend([
            'TransactionHour', 'TransactionDayOfWeek', 'TransactionMonth', 'TransactionYear', 'time_since_signup'
        ])

        # From TransactionFrequencyVelocity
        for id_col in self.id_cols_for_agg_after_rename:
            for window in [1, 7, 30]: # Assuming these are the windows used in TransactionFrequencyVelocity
                self.final_numerical_features.append(f'{id_col}_transactions_last_{window}d')
                self.final_numerical_features.append(f'{id_col}_total_amount_last_{window}d')
                self.final_numerical_features.append(f'{id_col}_avg_amount_last_{window}d')

        self.final_numerical_features = list(set(self.final_numerical_features)) # Remove duplicates
        self.final_categorical_features = list(set(self.categorical_cols_after_merge))

        pipeline_steps.append(('final_feature_processor', FinalFeatureProcessor(
            numerical_cols_to_scale=self.final_numerical_features,
            categorical_cols_to_encode=self.final_categorical_features
        )))

        return Pipeline(steps=pipeline_steps)

    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)


class CreditCardDataProcessor:
    """
    A specific data processor for creditcard.csv, orchestrating relevant transformers.
    """
    def __init__(self, numerical_cols: list):
        self.numerical_cols = numerical_cols
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """
        Builds the scikit-learn pipeline for creditcard.csv preprocessing.
        """
        pipeline_steps = [
            ('remove_duplicates', RemoveDuplicates()),
            ('data_type_corrector', DataTypeCorrector(
                numerical_cols=self.numerical_cols
            )),
            ('handle_amount_features', HandleAmountFeatures(amount_col='Amount')),
            ('ensure_numerical_dtypes', EnsureNumericalDtypes()),
            ('final_feature_processor', FinalFeatureProcessor(
                numerical_cols_to_scale=[col for col in self.numerical_cols if col != 'Time'] + ['IsRefund', 'Time'], # Ensure Time is treated as numerical and scaled
                categorical_cols_to_encode=[] # Credit card data has no explicit categorical features
            ))
        ]
        return Pipeline(steps=pipeline_steps)

    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)


# Example usage for independent testing (remains CPU-based for simplicity)
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    fraud_data_path = project_root / "data" / "raw" / "Fraud_Data.csv"
    creditcard_data_path = project_root / "data" / "raw" / "creditcard.csv"
    ip_to_country_path = project_root / "data" / "raw" / "IpAddress_to_Country.csv"

    print("--- Testing FraudDataProcessor (CPU-only for example) ---")
    fraud_df_raw = load_data(fraud_data_path)
    ip_country_df_test = load_data(ip_to_country_path)

    if not fraud_df_raw.empty:
        fraud_target_col = 'class'
        X_fraud = fraud_df_raw.drop(columns=[fraud_target_col])
        y_fraud = fraud_df_raw[fraud_target_col]

        # Simulate renaming and IP merge
        X_fraud_simulated_renamed = X_fraud.rename(columns={
            'user_id': 'CustomerId',
            'purchase_value': 'Amount',
            'purchase_time': 'TransactionStartTime',
            'user_id': 'TransactionId'
        }).copy()
        from src.utils.helpers import merge_ip_to_country # Ensure this is imported
        X_fraud_simulated_renamed = merge_ip_to_country(X_fraud_simulated_renamed, ip_country_df_test.copy())

        fraud_numerical_features_renamed = ['Amount', 'age']
        fraud_categorical_features_renamed = ['source', 'browser', 'sex', 'country']
        fraud_purchase_time_col_renamed = 'TransactionStartTime'
        fraud_signup_time_col_renamed = 'signup_time'
        fraud_amount_col_renamed = 'Amount'
        fraud_id_cols_for_agg_renamed = ['CustomerId', 'device_id', 'ip_address']

        fraud_processor = FraudDataProcessor(
            numerical_cols_after_rename=fraud_numerical_features_renamed,
            categorical_cols_after_merge=fraud_categorical_features_renamed,
            time_col_after_rename=fraud_purchase_time_col_renamed,
            signup_time_col_after_rename=fraud_signup_time_col_renamed,
            amount_col_after_rename=fraud_amount_col_renamed,
            id_cols_for_agg_after_rename=fraud_id_cols_for_agg_renamed
        )
        
        X_fraud_processed = fraud_processor.fit_transform(X_fraud_simulated_renamed, y_fraud)

        print("\nProcessed Fraud_Data.csv head:")
        print(X_fraud_processed.head())
        print("\nProcessed Fraud_Data.csv info:")
        X_fraud_processed.info()
        print("\nMissing values in processed Fraud_Data.csv:")
        print(X_fraud_processed.isnull().sum()[X_fraud_processed.isnull().sum() > 0])
        print("\nVerifying new features in Fraud_Data.csv:")
        if 'IsRefund' in X_fraud_processed.columns:
            print(f"IsRefund value counts:\n{X_fraud_processed['IsRefund'].value_counts()}")
        if 'TransactionHour' in X_fraud_processed.columns:
            print(f"TransactionHour value counts:\n{X_fraud_processed['TransactionHour'].value_counts().head()}")
        if 'CustomerId_transactions_last_7d' in X_fraud_processed.columns:
            print(f"CustomerId_transactions_last_7d head:\n{X_fraud_processed['CustomerId_transactions_last_7d'].head()}")
    else:
        print("Fraud_Data.csv is empty. Skipping FraudDataProcessor test.")

    print("\n" + "="*50 + "\n")

    print("--- Testing CreditCardDataProcessor (CPU-only for example) ---")
    creditcard_df_raw = load_data(creditcard_data_path)

    if not creditcard_df_raw.empty:
        creditcard_target_col = 'Class'
        if creditcard_target_col in creditcard_df_raw.columns:
            X_creditcard = creditcard_df_raw.drop(columns=[creditcard_target_col])
            y_creditcard = creditcard_df_raw[creditcard_target_col]
        else:
            print(f"Warning: Target column '{creditcard_target_col}' not found in creditcard.csv. Using a dummy target.")
            X_creditcard = creditcard_df_raw.copy()
            y_creditcard = pd.Series([0] * len(X_creditcard), index=X_creditcard.index)

        creditcard_numerical_features = [col for col in X_creditcard.columns if col not in []] # All columns are numerical except target

        creditcard_processor = CreditCardDataProcessor(
            numerical_cols=creditcard_numerical_features
        )

        X_creditcard_processed = creditcard_processor.fit_transform(X_creditcard, y_creditcard)

        print("\nProcessed creditcard.csv head:")
        print(X_creditcard_processed.head())
        print("\nProcessed creditcard.csv info:")
        X_creditcard_processed.info()
        print("\nMissing values in processed creditcard.csv:")
        print(X_creditcard_processed.isnull().sum()[X_creditcard_processed.isnull().sum() > 0])
        print("\nVerifying new features in creditcard.csv:")
        if 'IsRefund' in X_creditcard_processed.columns:
            print(f"IsRefund value counts:\n{X_creditcard_processed['IsRefund'].value_counts()}")
    else:
        print("creditcard.csv is empty. Skipping CreditCardDataProcessor test.")
