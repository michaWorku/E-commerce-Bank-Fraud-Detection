import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import datetime


class HandleAmountFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle transaction 'Amount' values:
    1. Creates an 'IsRefund' indicator for originally negative amounts.
    2. Converts negative amounts to positive (absolute value).
    Ensures 'Amount' is numeric and handles NaNs.
    """
    def __init__(self, amount_col: str = 'Amount'):
        self.amount_col = amount_col
        self.is_refund_col = 'IsRefund'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if X_copy.empty:
            print("HandleAmountFeatures: Input DataFrame is empty. Returning empty DataFrame.")
            empty_df = X_copy.copy()
            empty_df[self.is_refund_col] = [] 
            return empty_df

        if self.amount_col in X_copy.columns:
            X_copy[self.amount_col] = pd.to_numeric(X_copy[self.amount_col], errors='coerce').fillna(0.0).astype(float)
            
            X_copy[self.is_refund_col] = (X_copy[self.amount_col] < 0).astype(int)
            
            X_copy[self.amount_col] = X_copy[self.amount_col].abs()
        else:
            print(f"Warning: Amount column '{self.amount_col}' not found. Skipping amount feature handling.")
            X_copy[self.is_refund_col] = 0 

        return X_copy


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer temporal features from transaction and signup timestamps.
    - Extracts hour, day of week, month, year from purchase time.
    - Calculates time since signup in days.
    Ensures datetime columns are correctly parsed and handles NaNs.
    """
    def __init__(self, purchase_time_col: str = 'TransactionStartTime', signup_time_col: str = 'signup_time'):
        self.purchase_time_col = purchase_time_col
        self.signup_time_col = signup_time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if X_copy.empty:
            print("TemporalFeatureEngineer: Input DataFrame is empty. Returning empty DataFrame.")
            empty_df = X_copy.copy()
            empty_df['TransactionHour'] = []
            empty_df['TransactionDayOfWeek'] = []
            empty_df['TransactionMonth'] = []
            empty_df['TransactionYear'] = []
            empty_df['time_since_signup'] = []
            return empty_df

        if self.purchase_time_col in X_copy.columns:
            X_copy[self.purchase_time_col] = pd.to_datetime(X_copy[self.purchase_time_col], errors='coerce', utc=True)
            
            X_copy['TransactionHour'] = X_copy[self.purchase_time_col].dt.hour.fillna(0).astype(int)
            X_copy['TransactionDayOfWeek'] = X_copy[self.purchase_time_col].dt.dayofweek.fillna(0).astype(int)
            X_copy['TransactionMonth'] = X_copy[self.purchase_time_col].dt.month.fillna(0).astype(int)
            X_copy['TransactionYear'] = X_copy[self.purchase_time_col].dt.year.fillna(0).astype(int)
        else:
            print(f"Warning: Purchase time column '{self.purchase_time_col}' not found. Skipping temporal feature extraction.")
            X_copy['TransactionHour'] = 0
            X_copy['TransactionDayOfWeek'] = 0
            X_copy['TransactionMonth'] = 0
            X_copy['TransactionYear'] = 0

        if self.signup_time_col in X_copy.columns and self.purchase_time_col in X_copy.columns:
            X_copy[self.signup_time_col] = pd.to_datetime(X_copy[self.signup_time_col], errors='coerce', utc=True)

            time_diff = (X_copy[self.purchase_time_col] - X_copy[self.signup_time_col]).dt.total_seconds() / (24 * 3600)
            
            X_copy['time_since_signup'] = pd.to_numeric(time_diff, errors='coerce').fillna(0.0).apply(lambda x: x if x >= 0 else 0).astype(float)
        else:
            print(f"Warning: Signup time column '{self.signup_time_col}' or purchase time column '{self.purchase_time_col}' not found. Skipping time since signup calculation.")
            X_copy['time_since_signup'] = 0.0 

        return X_copy


class TransactionFrequencyVelocity(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate transaction frequency and amount velocity features
    for specified ID columns (e.g., CustomerId, DeviceId, IpAddress).
    Calculates:
    - Number of transactions in last N days
    - Total amount in last N days
    - Average amount in last N days
    """
    def __init__(self, id_cols: list, time_col: str = 'TransactionStartTime', amount_col: str = 'Amount', time_windows_days: list = None):
        self.id_cols = id_cols
        self.time_col = time_col
        self.amount_col = amount_col
        self.time_windows_days = time_windows_days if time_windows_days is not None else [1, 7, 30]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if X_copy.empty:
            print("TransactionFrequencyVelocity: Input DataFrame is empty. Returning empty DataFrame.")
            for id_col in self.id_cols:
                for window in self.time_windows_days:
                    X_copy[f'{id_col}_transactions_last_{window}d'] = 0
                    X_copy[f'{id_col}_total_amount_last_{window}d'] = 0.0
                    X_copy[f'{id_col}_avg_amount_last_{window}d'] = 0.0
            return X_copy

        if self.time_col not in X_copy.columns:
            print(f"Error: Time column '{self.time_col}' not found for TransactionFrequencyVelocity.")
            return X_copy
        if self.amount_col not in X_copy.columns:
            print(f"Error: Amount column '{self.amount_col}' not found for TransactionFrequencyVelocity.")
            X_copy[self.amount_col] = 0.0 
            print(f"Created dummy '{self.amount_col}' column with 0.0 for calculations.")

        X_copy[self.time_col] = pd.to_datetime(X_copy[self.time_col], errors='coerce', utc=True)
        X_copy[self.amount_col] = pd.to_numeric(X_copy[self.amount_col], errors='coerce').fillna(0.0).astype(float)

        # Create a temporary unique ID for each row to ensure stable merging
        temp_unique_id_col = '__temp_unique_id__'
        X_copy[temp_unique_id_col] = X_copy.index # Use original index as unique ID

        # Sort by time_col and then by the temporary unique ID for stable order
        X_copy_sorted = X_copy.dropna(subset=[self.time_col]).sort_values(by=[self.time_col, temp_unique_id_col]).copy()
        
        existing_id_cols = [col for col in self.id_cols if col in X_copy_sorted.columns]
        if not existing_id_cols:
            print("Warning: No valid ID columns found for transaction frequency/velocity calculation.")
            # Ensure new columns are added to X_copy even if no ID columns exist
            for id_col in self.id_cols:
                for window in self.time_windows_days:
                    X_copy[f'{id_col}_transactions_last_{window}d'] = 0
                    X_copy[f'{id_col}_total_amount_last_{window}d'] = 0.0
                    X_copy[f'{id_col}_avg_amount_last_{window}d'] = 0.0
            X_copy = X_copy.drop(columns=[temp_unique_id_col]) # Drop temp column before returning
            return X_copy

        # Initialize a DataFrame to store new features, indexed by the temporary unique ID
        new_features_data = {}

        for id_col in existing_id_cols:
            grouped = X_copy_sorted.groupby(id_col)
            
            for window in self.time_windows_days:
                # Define rolling window properties
                # We need to ensure the rolling operation is performed on the time index
                # and the result is re-indexed to the original temporary unique ID.
                
                # Create a temporary Series with time_col as index for rolling
                temp_series_for_rolling = X_copy_sorted.set_index(self.time_col)
                
                # Apply rolling operations within each group
                # The result will be a Series with a MultiIndex (id_col, time_col)
                rolling_counts = grouped.apply(
                    lambda x: x.set_index(self.time_col)[self.amount_col].rolling(f'{window}d', closed='left').count()
                )
                rolling_sums = grouped.apply(
                    lambda x: x.set_index(self.time_col)[self.amount_col].rolling(f'{window}d', closed='left').sum()
                )
                rolling_means = grouped.apply(
                    lambda x: x.set_index(self.time_col)[self.amount_col].rolling(f'{window}d', closed='left').mean()
                )

                # Now, map these results back to the original unique IDs in X_copy_sorted
                # We need to ensure the index of rolling_counts/sums/means matches the index of X_copy_sorted
                # The index of rolling_counts is (id_col, time_col). We need to get back to the temp_unique_id_col.
                
                # Reset index of rolling results to get id_col and time_col back as columns
                rolling_counts = rolling_counts.reset_index()
                rolling_sums = rolling_sums.reset_index()
                rolling_means = rolling_means.reset_index()

                # Merge rolling results back to X_copy_sorted using id_col and time_col
                # Then, extract the values corresponding to temp_unique_id_col in X_copy_sorted
                
                # Create a temporary DataFrame for merging rolling results
                rolling_results_df = pd.DataFrame({
                    'id_col_temp': rolling_counts[id_col],
                    'time_col_temp': rolling_counts[self.time_col],
                    'count': rolling_counts[self.amount_col],
                    'sum': rolling_sums[self.amount_col],
                    'mean': rolling_means[self.amount_col]
                })
                
                # Merge rolling results with X_copy_sorted based on id_col and time_col
                # This ensures correct alignment even with duplicate timestamps, as temp_unique_id_col is preserved
                merged_rolling = pd.merge(
                    X_copy_sorted[[id_col, self.time_col, temp_unique_id_col]], # Only necessary columns for merge
                    rolling_results_df,
                    left_on=[id_col, self.time_col],
                    right_on=['id_col_temp', 'time_col_temp'],
                    how='left'
                )
                
                # Assign to new_features_data dictionary, indexed by temp_unique_id_col
                new_features_data[f'{id_col}_transactions_last_{window}d'] = merged_rolling.set_index(temp_unique_id_col)['count'].fillna(0).astype(int)
                new_features_data[f'{id_col}_total_amount_last_{window}d'] = merged_rolling.set_index(temp_unique_id_col)['sum'].fillna(0.0).astype(float)
                new_features_data[f'{id_col}_avg_amount_last_{window}d'] = merged_rolling.set_index(temp_unique_id_col)['mean'].fillna(0.0).astype(float)
        
        # Convert new_features_data dictionary to a DataFrame
        new_features_df_final = pd.DataFrame(new_features_data, index=X_copy[temp_unique_id_col])
        
        # Merge the new features back to the original X_copy using the temporary unique ID
        X_copy = X_copy.merge(new_features_df_final, left_on=temp_unique_id_col, right_index=True, how='left')

        # Drop the temporary unique ID column
        X_copy = X_copy.drop(columns=[temp_unique_id_col])

        return X_copy
