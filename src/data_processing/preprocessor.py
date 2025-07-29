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

# Import from the new loader path (if needed for internal testing)
from src.data_processing.loader import load_data

# Import feature engineering transformers from the new module
# CRITICAL FIX: Add import for HandleAmountFeatures, TemporalFeatureEngineer, TransactionFrequencyVelocity
from src.feature_engineering.engineer import HandleAmountFeatures, TemporalFeatureEngineer, TransactionFrequencyVelocity


# Attempt to import cudf for GPU acceleration
try:
    import cudf
    _CUDF_AVAILABLE = True
    print("cuDF is available in preprocessor.py. Transformers can use GPU.")
except ImportError:
    _CUDF_AVAILABLE = False
    print("cuDF not available in preprocessor.py. Falling back to pandas (CPU).")


# Custom transformer to convert cuDF DataFrame to pandas DataFrame
class CudfToPandas(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if _CUDF_AVAILABLE and isinstance(X, cudf.DataFrame):
            print("Converting DataFrame from cuDF to pandas...")
            return X.to_pandas()
        return X

# Custom transformer to convert pandas DataFrame back to cuDF DataFrame
class PandasToCudf(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if _CUDF_AVAILABLE and isinstance(X, pd.DataFrame):
            print("Converting DataFrame from pandas to cuDF...")
            return cudf.DataFrame.from_pandas(X)
        return X


class RemoveDuplicates(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove duplicate rows from a DataFrame.
    Supports both pandas and cuDF DataFrames.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_cudf = _CUDF_AVAILABLE and isinstance(X, cudf.DataFrame)
        df_lib = cudf if is_cudf else pd

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
    Optimized to apply conversions in a vectorized manner where possible.
    Supports both pandas and cuDF DataFrames.
    """
    def __init__(self, datetime_cols: list = None, numerical_cols: list = None):
        self.datetime_cols = datetime_cols if datetime_cols is not None else []
        self.numerical_cols = numerical_cols if numerical_cols is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_cudf = _CUDF_AVAILABLE and isinstance(X, cudf.DataFrame)
        
        X_copy = X.copy()
        if X_copy.empty:
            print("DataTypeCorrector: Input DataFrame is empty. Returning empty DataFrame.")
            return X_copy

        # Convert specified columns to datetime
        for col in self.datetime_cols:
            if col in X_copy.columns:
                print(f"Converting '{col}' to datetime using {'cuDF' if is_cudf else 'pandas'}...")
                if is_cudf:
                    # Workaround for cuDF's to_datetime not supporting errors='coerce' directly on Series
                    # Convert to pandas Series, then to datetime, then back to cuDF Series
                    X_copy[col] = cudf.Series(pd.to_datetime(X_copy[col].to_pandas(), errors='coerce', utc=True))
                else:
                    X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce', utc=True)
            else:
                print(f"Warning: Datetime column '{col}' not found for type correction.")

        # Convert specified columns to numerical
        for col in self.numerical_cols:
            if col in X_copy.columns:
                print(f"Converting '{col}' to numerical using {'cuDF' if is_cudf else 'pandas'}...")
                if is_cudf:
                    # CRITICAL FIX: Ensure to_numeric, fillna, and astype for cuDF
                    X_copy[col] = cudf.to_numeric(X_copy[col], errors='coerce').fillna(0.0).astype(float)
                else:
                    # CRITICAL FIX: Ensure to_numeric, fillna, and astype for pandas
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0.0).astype(float)
            else:
                print(f"Warning: Numerical column '{col}' not found for type correction.")
        
        return X_copy


class FinalFeatureProcessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies final imputation, scaling of numerical features,
    and One-Hot Encoding of categorical features.
    It handles conversion between cuDF and pandas DataFrames for scikit-learn compatibility.
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
        is_cudf_input = _CUDF_AVAILABLE and isinstance(X, cudf.DataFrame)
        
        if X.empty:
            print("FinalFeatureProcessor: Input DataFrame is empty. Skipping fit.")
            self.fitted_feature_names_out = [] # No features if empty
            return self

        # Convert to pandas for scikit-learn's ColumnTransformer fit
        X_pd = X.to_pandas() if is_cudf_input else X.copy()
        y_pd = y.to_pandas() if is_cudf_input and y is not None else y

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
            self.fitted_feature_names_out = self.preprocessor.get_feature_names_out()

        return self

    def transform(self, X):
        is_cudf_input = _CUDF_AVAILABLE and isinstance(X, cudf.DataFrame)

        if X.empty:
            print("FinalFeatureProcessor: Input DataFrame is empty. Returning empty DataFrame.")
            if self.fitted_feature_names_out is not None:
                # Return empty DataFrame of the correct type (cudf or pandas)
                return cudf.DataFrame(columns=self.fitted_feature_names_out) if is_cudf_input else pd.DataFrame(columns=self.fitted_feature_names_out)
            else:
                return X.copy()
        
        if self.preprocessor is None: # Case where fit was called on an empty dataframe
            print("FinalFeatureProcessor: Transformer not fitted. Returning original DataFrame.")
            return X.copy()
        
        if self.preprocessor == 'passthrough':
            return X.copy()

        # Convert to pandas for scikit-learn's ColumnTransformer transform
        X_pd = X.to_pandas() if is_cudf_input else X.copy()

        X_transformed_array = self.preprocessor.transform(X_pd)

        # Create DataFrame from transformed array, using fitted feature names
        if is_cudf_input:
            X_transformed_df = cudf.DataFrame(X_transformed_array, columns=self.fitted_feature_names_out, index=X.index)
        else:
            X_transformed_df = pd.DataFrame(X_transformed_array, columns=self.fitted_feature_names_out, index=X.index)
        
        # CRITICAL FIX: Explicitly cast all numerical columns to float after transformation.
        # This is a robust final step to ensure correct dtypes for downstream tasks (like EDA).
        for col in self.numerical_cols_to_scale: # Iterate over the columns that *should* be numerical
            if col in X_transformed_df.columns:
                if is_cudf_input:
                    X_transformed_df[col] = cudf.to_numeric(X_transformed_df[col], errors='coerce').fillna(0.0).astype(float)
                else:
                    X_transformed_df[col] = pd.to_numeric(X_transformed_df[col], errors='coerce').fillna(0.0).astype(float)
        
        return X_transformed_df


class FraudDataProcessor:
    """
    A specific data processor for Fraud_Data.csv, orchestrating relevant transformers.
    It expects column names *after* initial renaming (e.g., 'Amount' instead of 'purchase_value').
    Handles both pandas and cuDF DataFrames.
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
            # Conditionally convert to pandas before TransactionFrequencyVelocity
            ('cudf_to_pandas_for_rolling', CudfToPandas()),
            ('transaction_frequency_velocity', TransactionFrequencyVelocity(
                id_cols=self.id_cols_for_agg_after_rename,
                time_col=self.time_col_after_rename,
                amount_col=self.amount_col_after_rename
            )),
            # Conditionally convert back to cuDF after TransactionFrequencyVelocity
            ('pandas_to_cudf_after_rolling', PandasToCudf()),
            ('temporal_feature_engineer', TemporalFeatureEngineer(
                purchase_time_col=self.time_col_after_rename,
                signup_time_col=self.signup_time_col_after_rename
            ))
        ]
        
        self.final_numerical_features = list(self.numerical_cols_after_rename)
        self.final_numerical_features.append('IsRefund')

        for id_col in self.id_cols_for_agg_after_rename:
            for window in [1, 7, 30]:
                self.final_numerical_features.append(f'{id_col}_transactions_last_{window}d')
                self.final_numerical_features.append(f'{id_col}_total_amount_last_{window}d')
                self.final_numerical_features.append(f'{id_col}_avg_amount_last_{window}d')

        self.final_numerical_features.extend([
            'TransactionHour', 'TransactionDayOfWeek', 'TransactionMonth', 'TransactionYear', 'time_since_signup'
        ])
        
        self.final_numerical_features = list(set(self.final_numerical_features))
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
    Handles both pandas and cuDF DataFrames.
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
            # For creditcard data, if it also needs complex rolling features, similar wrappers would be needed
            # For now, assuming it doesn't need complex rolling or that 'Time' column is handled differently
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
    fraud_df_raw = load_data(fraud_data_path, use_gpu=False) # Force CPU for example
    ip_country_df_test = load_data(ip_to_country_path, use_gpu=False) # Force CPU for example

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
    else:
        print("Fraud_Data.csv is empty. Skipping FraudDataProcessor test.")
