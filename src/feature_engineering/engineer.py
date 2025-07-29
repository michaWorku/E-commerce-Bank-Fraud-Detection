import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import datetime

# Attempt to import cudf for GPU acceleration
try:
    import cudf
    _CUDF_AVAILABLE = True
    print("cuDF is available in engineer.py. Transformers can use GPU.")
except ImportError:
    _CUDF_AVAILABLE = False
    print("cuDF not available in engineer.py. Falling back to pandas (CPU).")


class HandleAmountFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle transaction 'Amount' values:
    1. Creates an 'IsRefund' indicator for originally negative amounts.
    2. Converts negative amounts to positive (absolute value).
    Supports both pandas and cuDF DataFrames.
    """
    def __init__(self, amount_col: str = 'Amount'):
        self.amount_col = amount_col
        self.is_refund_col = 'IsRefund'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_cudf = _CUDF_AVAILABLE and isinstance(X, cudf.DataFrame)
        
        X_copy = X.copy()
        if X_copy.empty:
            print("HandleAmountFeatures: Input DataFrame is empty. Returning empty DataFrame.")
            empty_df = X_copy.copy()
            empty_df[self.is_refund_col] = []
            return empty_df

        if self.amount_col in X_copy.columns:
            # Use cuDF's to_numeric if available, else pandas
            if is_cudf:
                X_copy[self.amount_col] = cudf.to_numeric(X_copy[self.amount_col], errors='coerce')
            else:
                X_copy[self.amount_col] = pd.to_numeric(X_copy[self.amount_col], errors='coerce')
            
            # Ensure 'IsRefund' is created with the correct DataFrame type
            if is_cudf:
                X_copy[self.is_refund_col] = (X_copy[self.amount_col] < 0).astype('int32') # Use int32 for cuDF
            else:
                X_copy[self.is_refund_col] = (X_copy[self.amount_col] < 0).astype(int)
            
            X_copy[self.amount_col] = X_copy[self.amount_col].abs()
        else:
            print(f"Warning: '{self.amount_col}' column not found for HandleAmountFeatures. Skipping transformation and adding default 'IsRefund'.")
            if is_cudf:
                X_copy[self.is_refund_col] = cudf.Series([0] * len(X_copy), index=X_copy.index, dtype='int32')
            else:
                X_copy[self.is_refund_col] = 0
        return X_copy


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract temporal features from datetime columns
    and calculate 'time_since_signup'.
    Supports both pandas and cuDF DataFrames.
    """
    def __init__(self, purchase_time_col: str = 'TransactionStartTime', signup_time_col: str = 'signup_time'):
        self.purchase_time_col = purchase_time_col
        self.signup_time_col = signup_time_col
        self.extracted_features = [
            'TransactionHour', 'TransactionDayOfWeek', 'TransactionMonth', 'TransactionYear',
            'time_since_signup'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_cudf = _CUDF_AVAILABLE and isinstance(X, cudf.DataFrame)
        
        X_copy = X.copy()
        if X_copy.empty:
            print("TemporalFeatureEngineer: Input DataFrame is empty. Returning empty DataFrame.")
            empty_df = X_copy.copy()
            for feat in self.extracted_features:
                empty_df[feat] = np.nan
            return empty_df

        for col in [self.purchase_time_col, self.signup_time_col]:
            if col in X_copy.columns:
                print(f"Converting '{col}' to datetime using {'cuDF' if is_cudf else 'pandas'}...")
                if is_cudf:
                    # Workaround for cuDF's to_datetime not supporting errors='coerce' on Series
                    X_copy[col] = cudf.Series(pd.to_datetime(X_copy[col].to_pandas(), errors='coerce', utc=True))
                else:
                    X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce', utc=True)

        if self.purchase_time_col in X_copy.columns:
            initial_len = len(X_copy)
            X_copy.dropna(subset=[self.purchase_time_col], inplace=True)
            if len(X_copy) < initial_len:
                print(f"TemporalFeatureEngineer: Dropped {initial_len - len(X_copy)} rows due to NaT in '{self.purchase_time_col}'.")

            if X_copy.empty:
                print(f"Warning: DataFrame became empty after dropping NaNs in '{self.purchase_time_col}'. Returning empty DataFrame.")
                empty_df = X_copy.copy()
                for feat in self.extracted_features:
                    empty_df[feat] = np.nan
                return empty_df

            X_copy['TransactionHour'] = X_copy[self.purchase_time_col].dt.hour
            X_copy['TransactionDayOfWeek'] = X_copy[self.purchase_time_col].dt.dayofweek
            X_copy['TransactionMonth'] = X_copy[self.purchase_time_col].dt.month
            X_copy['TransactionYear'] = X_copy[self.purchase_time_col].dt.year
        else:
            print(f"Warning: Purchase time column '{self.purchase_time_col}' not found. Skipping related temporal feature extraction.")
            for col in ['TransactionHour', 'TransactionDayOfWeek', 'TransactionMonth', 'TransactionYear']:
                if is_cudf:
                    X_copy[col] = cudf.Series(np.nan, index=X_copy.index, dtype='float64')
                else:
                    X_copy[col] = np.nan

        if self.purchase_time_col in X_copy.columns and self.signup_time_col in X_copy.columns:
            time_diff = (X_copy[self.purchase_time_col] - X_copy[self.signup_time_col]).dt.total_seconds() / (24 * 3600)
            
            if is_cudf:
                # Fillna with median, ensuring it's a cuDF Series
                median_val = time_diff.median()
                X_copy['time_since_signup'] = time_diff.fillna(median_val)
                # Apply lambda for non-negative, converting to pandas then back
                X_copy['time_since_signup'] = cudf.Series(X_copy['time_since_signup'].to_pandas().apply(lambda x: x if x >= 0 else 0))
            else:
                X_copy['time_since_signup'] = time_diff.fillna(time_diff.median())
                X_copy['time_since_signup'] = X_copy['time_since_signup'].apply(lambda x: x if x >= 0 else 0)
        else:
            print(f"Warning: Cannot calculate 'time_since_signup'. Missing '{self.purchase_time_col}' or '{self.signup_time_col}'.")
            if is_cudf:
                X_copy['time_since_signup'] = cudf.Series(np.nan, index=X_copy.index, dtype='float64')
            else:
                X_copy['time_since_signup'] = np.nan
        return X_copy


class TransactionFrequencyVelocity(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate transaction frequency and velocity features.
    This transformer is designed to work with pandas DataFrames.
    It expects to receive a pandas DataFrame and will return a pandas DataFrame.
    The conversion from/to cuDF should be handled upstream in the pipeline.
    """
    def __init__(self, id_cols: list, time_col: str, amount_col: str, time_windows_days: list = None):
        self.id_cols = id_cols
        self.time_col = time_col
        self.amount_col = amount_col
        self.time_windows_days = time_windows_days if time_windows_days is not None else [1, 7, 30]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Explicitly convert to pandas DataFrame at the entry point of this transformer
        # This ensures all subsequent operations are pure pandas.
        X_copy = X.to_pandas() if _CUDF_AVAILABLE and isinstance(X, cudf.DataFrame) else X.copy()

        if X_copy.empty:
            print("TransactionFrequencyVelocity: Input DataFrame is empty. Returning empty DataFrame.")
            return X_copy

        if self.time_col not in X_copy.columns:
            print(f"Warning: Time column '{self.time_col}' not found. Skipping TransactionFrequencyVelocity.")
            return X_copy
        
        # Ensure time column is datetime (pandas operation)
        X_copy[self.time_col] = pd.to_datetime(X_copy[self.time_col], errors='coerce', utc=True)

        if self.amount_col in X_copy.columns:
            X_copy[self.amount_col] = pd.to_numeric(X_copy[self.amount_col], errors='coerce')
        else:
            print(f"Warning: Amount column '{self.amount_col}' not found. Skipping amount-based velocity features.")

        existing_id_cols = [col for col in self.id_cols if col in X_copy.columns]
        if not existing_id_cols:
            print(f"Warning: None of the specified ID columns {self.id_cols} found. Skipping TransactionFrequencyVelocity.")
            return X_copy

        cols_to_check_for_nan = [self.time_col] + existing_id_cols
        initial_len = len(X_copy)
        X_copy.dropna(subset=cols_to_check_for_nan, inplace=True)
        if len(X_copy) < initial_len:
            print(f"TransactionFrequencyVelocity: Dropped {initial_len - len(X_copy)} rows due to NaNs in critical columns.")
        
        if X_copy.empty:
            print("Warning: DataFrame became empty after dropping NaNs for TransactionFrequencyVelocity. Returning empty DataFrame.")
            return X_copy

        original_index = X_copy.index # Store original index for re-alignment

        # Add a unique row ID to ensure a stable and unique sort order
        X_copy['_row_id'] = np.arange(len(X_copy))

        # Sort by ID, time, and then the unique row ID to guarantee strict monotonicity
        X_copy = X_copy.sort_values(by=existing_id_cols + [self.time_col, '_row_id']).reset_index(drop=True)
        
        # Helper function to apply rolling calculations to each group
        def _calculate_rolling_features_for_group(group_df, time_col, amount_col, time_windows_days):
            # Ensure the group's index is the time column and is monotonic
            # This is crucial for rolling operations within each group
            group_df_sorted = group_df.set_index(time_col, drop=False).sort_index()

            for window in time_windows_days:
                # Frequency: transactions count
                group_df_sorted[f'transactions_last_{window}d'] = \
                    group_df_sorted[time_col].rolling(f'{window}d', closed='left').count().fillna(0).astype(int)

                # Velocity: sum/mean of amount
                if amount_col in group_df_sorted.columns:
                    group_df_sorted[f'total_amount_last_{window}d'] = \
                        group_df_sorted[amount_col].rolling(f'{window}d', closed='left').sum().fillna(0)
                    group_df_sorted[f'avg_amount_last_{window}d'] = \
                        group_df_sorted[amount_col].rolling(f'{window}d', closed='left').mean().fillna(0)
                else:
                    group_df_sorted[f'total_amount_last_{window}d'] = 0.0
                    group_df_sorted[f'avg_amount_last_{window}d'] = 0.0
            
            # Reset index to return to original flat structure, but keep original order for merging
            return group_df_sorted.reset_index(drop=True)

        # Apply the rolling calculations group by group using `apply`
        # `group_keys=False` is important for the resulting index structure when applying to many groups
        calculated_features_df = X_copy.groupby(existing_id_cols, group_keys=False).apply(
            lambda group: _calculate_rolling_features_for_group(group, self.time_col, self.amount_col, self.time_windows_days)
        )
        
        # Merge the calculated features back to the original X_copy DataFrame using the temporary _row_id
        # We need to preserve the original index for scikit-learn pipeline alignment
        X_copy_final = X_copy.merge(
            calculated_features_df.drop(columns=existing_id_cols + [self.time_col, self.amount_col, '_temp_time_for_rolling'], errors='ignore'),
            left_on='_row_id',
            right_on='_row_id',
            how='left',
            suffixes=('', '_new_feature')
        )

        # Drop temporary columns
        X_copy_final.drop(columns=['_temp_time_for_rolling', '_row_id'], inplace=True, errors='ignore')

        # Re-align with the original DataFrame's index to restore original row order
        X_copy_final.index = original_index
        return X_copy_final
