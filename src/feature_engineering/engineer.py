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
            # Add placeholder columns to an empty DataFrame to maintain schema consistency
            empty_df = X_copy.copy()
            empty_df[self.is_refund_col] = [] # Ensure it's an empty list for consistent dtype if possible
            return empty_df

        if self.amount_col in X_copy.columns:
            # Ensure Amount is numeric before comparison and operations
            # CRITICAL FIX: Ensure to_numeric, fillna, and astype
            if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                X_copy[self.amount_col] = cudf.to_numeric(X_copy[self.amount_col], errors='coerce').fillna(0.0).astype(float)
            else:
                X_copy[self.amount_col] = pd.to_numeric(X_copy[self.amount_col], errors='coerce').fillna(0.0).astype(float)
            
            # Create 'IsRefund' flag: 1 if Amount is negative, 0 otherwise
            # This must be done *before* taking the absolute value
            X_copy[self.is_refund_col] = (X_copy[self.amount_col] < 0).astype(int)
            
            # Convert negative amounts to positive (absolute value)
            X_copy[self.amount_col] = X_copy[self.amount_col].abs()
        else:
            print(f"Warning: Amount column '{self.amount_col}' not found. Skipping amount feature handling.")
            # If amount_col is missing, create IsRefund with default 0s
            X_copy[self.is_refund_col] = 0 # Default to 0 if amount_col is not present

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
            # Add placeholder columns to maintain schema consistency
            empty_df = X_copy.copy()
            empty_df['TransactionHour'] = []
            empty_df['TransactionDayOfWeek'] = []
            empty_df['TransactionMonth'] = []
            empty_df['TransactionYear'] = []
            empty_df['time_since_signup'] = []
            return empty_df

        # Ensure datetime columns are in datetime format
        # CRITICAL FIX: Ensure to_datetime, fillna, and astype for both pandas and cuDF
        if self.purchase_time_col in X_copy.columns:
            if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                X_copy[self.purchase_time_col] = cudf.Series(pd.to_datetime(X_copy[self.purchase_time_col].to_pandas(), errors='coerce', utc=True))
            else:
                X_copy[self.purchase_time_col] = pd.to_datetime(X_copy[self.purchase_time_col], errors='coerce', utc=True)
            
            # Extract temporal features
            # Fill NaNs from datetime conversion with a placeholder before extraction to avoid errors
            # Then, fill extracted features' NaNs with 0 or a suitable value
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
            if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                X_copy[self.signup_time_col] = cudf.Series(pd.to_datetime(X_copy[self.signup_time_col].to_pandas(), errors='coerce', utc=True))
            else:
                X_copy[self.signup_time_col] = pd.to_datetime(X_copy[self.signup_time_col], errors='coerce', utc=True)

            # Calculate time since signup in days
            # CRITICAL FIX: Ensure total_seconds() result is numeric and handle NaNs
            time_diff = (X_copy[self.purchase_time_col] - X_copy[self.signup_time_col]).dt.total_seconds() / (24 * 3600)
            
            if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                X_copy['time_since_signup'] = cudf.to_numeric(time_diff, errors='coerce').fillna(0.0).apply(lambda x: x if x >= 0 else 0).astype(float)
            else:
                X_copy['time_since_signup'] = pd.to_numeric(time_diff, errors='coerce').fillna(0.0).apply(lambda x: x if x >= 0 else 0).astype(float)
        else:
            print(f"Warning: Signup time column '{self.signup_time_col}' or purchase time column '{self.purchase_time_col}' not found. Skipping time since signup calculation.")
            X_copy['time_since_signup'] = 0.0 # Default to 0.0 if columns are missing

        # Ensure original time columns are kept for potential later use (e.g., plotting in EDA)
        # No explicit dropping of original time columns here.
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
            # Add placeholder columns to the empty DataFrame to maintain schema consistency
            for id_col in self.id_cols:
                for window in self.time_windows_days:
                    X_copy[f'{id_col}_transactions_last_{window}d'] = 0
                    X_copy[f'{id_col}_total_amount_last_{window}d'] = 0.0
                    X_copy[f'{id_col}_avg_amount_last_{window}d'] = 0.0
            return X_copy

        # Ensure time column is datetime and amount column is numeric
        if self.time_col not in X_copy.columns:
            print(f"Error: Time column '{self.time_col}' not found for TransactionFrequencyVelocity.")
            return X_copy
        if self.amount_col not in X_copy.columns:
            print(f"Error: Amount column '{self.amount_col}' not found for TransactionFrequencyVelocity.")
            # If amount column is missing, we can't calculate amount-based features, but can still do frequency.
            # For robustness, let's create a dummy amount column if missing
            X_copy[self.amount_col] = 0.0
            print(f"Created dummy '{self.amount_col}' column with 0.0 for calculations.")

        # Ensure time column is datetime and amount is numeric AFTER potential initial conversion
        if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
            X_copy[self.time_col] = cudf.Series(pd.to_datetime(X_copy[self.time_col].to_pandas(), errors='coerce', utc=True))
            X_copy[self.amount_col] = cudf.to_numeric(X_copy[self.amount_col], errors='coerce').fillna(0.0).astype(float)
        else:
            X_copy[self.time_col] = pd.to_datetime(X_copy[self.time_col], errors='coerce', utc=True)
            X_copy[self.amount_col] = pd.to_numeric(X_copy[self.amount_col], errors='coerce').fillna(0.0).astype(float)

        # Drop rows where time_col is NaT (Not a Time) as they cannot be used for rolling calculations
        # and sort by time for rolling window operations
        X_copy_sorted = X_copy.dropna(subset=[self.time_col]).sort_values(by=self.time_col).copy()
        
        if X_copy_sorted.empty:
            print("TransactionFrequencyVelocity: DataFrame is empty after dropping NaT from time column. Cannot calculate features.")
            # Add placeholder columns to the empty DataFrame to maintain schema consistency
            for id_col in self.id_cols:
                for window in self.time_windows_days:
                    X_copy[f'{id_col}_transactions_last_{window}d'] = 0
                    X_copy[f'{id_col}_total_amount_last_{window}d'] = 0.0
                    X_copy[f'{id_col}_avg_amount_last_{window}d'] = 0.0
            return X_copy

        # Set time_col as index for rolling operations
        X_copy_sorted.set_index(self.time_col, inplace=True)

        # Dictionary to store new features, keyed by original index for re-alignment
        new_features_dict = {} 

        existing_id_cols = [col for col in self.id_cols if col in X_copy_sorted.columns]
        if not existing_id_cols:
            print("Warning: No valid ID columns found for transaction frequency/velocity calculation.")
            # Ensure placeholder columns are created if no ID columns were found
            for id_col in self.id_cols:
                for window in self.time_windows_days:
                    X_copy[f'{id_col}_transactions_last_{window}d'] = 0
                    X_copy[f'{id_col}_total_amount_last_{window}d'] = 0.0
                    X_copy[f'{id_col}_avg_amount_last_{window}d'] = 0.0
            return X_copy

        for id_col in existing_id_cols:
            grouped = X_copy_sorted.groupby(id_col)
            for window in self.time_windows_days:
                # Use .transform() to ensure the output Series aligns with the original DataFrame's index
                # The result of transform is a Series, whose index matches the original DataFrame's index
                # after the groupby.rolling operation.
                
                # Transactions count
                col_name_freq = f'{id_col}_transactions_last_{window}d'
                new_features_dict[col_name_freq] = grouped.rolling(f'{window}d', closed='left')[self.amount_col].count().transform(lambda x: x)
                
                # Total amount
                col_name_sum = f'{id_col}_total_amount_last_{window}d'
                new_features_dict[col_name_sum] = grouped.rolling(f'{window}d', closed='left')[self.amount_col].sum().transform(lambda x: x)
                
                # Average amount
                col_name_avg = f'{id_col}_avg_amount_last_{window}d'
                new_features_dict[col_name_avg] = grouped.rolling(f'{window}d', closed='left')[self.amount_col].mean().transform(lambda x: x)

        # Create a DataFrame from the new features dictionary
        # The keys of new_features_dict are column names, values are Series.
        # Ensure the index of this new DataFrame matches the original X_copy's index.
        # The Series returned by .transform() will have a MultiIndex (id_col, time_col).
        # We need to reset this MultiIndex to get back to the original index for joining.
        
        # Consolidate new features into a single DataFrame that can be joined
        # This is the crucial part to avoid _new_feature columns.
        
        # Create a list of Series, then concatenate them to a DataFrame
        calculated_features_list = []
        for col_name, series in new_features_dict.items():
            # The series from .transform() on a groupby.rolling has a MultiIndex (id_col, time_col)
            # We need to get it back to the original index of X_copy_sorted
            # Use .droplevel(0) to remove the id_col level from the MultiIndex
            # Then align with the original index of X_copy_sorted
            aligned_series = series.droplevel(0).reindex(X_copy_sorted.index)
            calculated_features_list.append(aligned_series.rename(col_name))

        if not calculated_features_list:
            print("No features calculated by TransactionFrequencyVelocity. Returning original DataFrame.")
            return X_copy # Return original if no features were calculated

        if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
            # Convert to pandas first for concatenation, then back to cuDF
            calculated_features_df = pd.concat([s.to_pandas() if isinstance(s, cudf.Series) else s for s in calculated_features_list], axis=1)
            calculated_features_df = cudf.DataFrame.from_pandas(calculated_features_df)
        else:
            calculated_features_df = pd.concat(calculated_features_list, axis=1)

        # Reset the index of X_copy_sorted to merge back correctly
        X_copy_sorted.reset_index(inplace=True) # Reset time_col index
        
        # Now, merge calculated_features_df back to the original X_copy using the original index
        # We need to align calculated_features_df's index with X_copy's index
        
        # The calculated_features_df has the time_col as index (from .transform() and .droplevel(0).reindex())
        # We need to merge it back based on the original index of X_copy.
        # Let's ensure calculated_features_df has the original index of X_copy_sorted before the set_index(self.time_col)
        
        # Simpler approach: add new features as columns to X_copy directly by aligning on index
        # The series in new_features_dict already have the correct index from .transform()
        # So we can just iterate and add them directly to X_copy
        
        # Reset index of X_copy_sorted to match X_copy's original index
        X_copy_sorted.index = X_copy.index[X_copy.index.isin(X_copy_sorted.index)] # Align indices
        
        # Add the new features back to X_copy
        for col_name, series in new_features_dict.items():
            # Ensure the series index aligns with X_copy's index
            if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                # For cuDF, reindex is needed to align the series with the original DataFrame's index
                X_copy[col_name] = series.reindex(X_copy.index).fillna(0.0) # Fillna for rows that might not have a rolling value
            else:
                X_copy[col_name] = series.reindex(X_copy.index).fillna(0.0) # Fillna for rows that might not have a rolling value
        
        # Fill any NaNs that might have been introduced for new features (e.g., if a group was empty before transformation)
        for id_col in existing_id_cols:
            for window in self.time_windows_days:
                col_freq = f'{id_col}_transactions_last_{window}d'
                col_sum = f'{id_col}_total_amount_last_{window}d'
                col_avg = f'{id_col}_avg_amount_last_{window}d'
                
                # Fill NaNs with 0 for new features and ensure correct dtype
                if col_freq in X_copy.columns:
                    if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                        X_copy[col_freq] = X_copy[col_freq].fillna(0).astype('int32')
                    else:
                        X_copy[col_freq] = X_copy[col_freq].fillna(0).astype(int)
                else: # If column was not created, create it with zeros
                    if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                        X_copy[col_freq] = cudf.Series(0, index=X_copy.index, dtype='int32')
                    else:
                        X_copy[col_freq] = pd.Series(0, index=X_copy.index, dtype=int)

                if col_sum in X_copy.columns:
                    if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                        X_copy[col_sum] = X_copy[col_sum].fillna(0.0).astype(float)
                    else:
                        X_copy[col_sum] = X_copy[col_sum].fillna(0.0).astype(float)
                else:
                    if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                        X_copy[col_sum] = cudf.Series(0.0, index=X_copy.index, dtype=float)
                    else:
                        X_copy[col_sum] = pd.Series(0.0, index=X_copy.index, dtype=float)

                if col_avg in X_copy.columns:
                    if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                        X_copy[col_avg] = X_copy[col_avg].fillna(0.0).astype(float)
                    else:
                        X_copy[col_avg] = X_copy[col_avg].fillna(0.0).astype(float)
                else:
                    if _CUDF_AVAILABLE and isinstance(X_copy, cudf.DataFrame):
                        X_copy[col_avg] = cudf.Series(0.0, index=X_copy.index, dtype=float)
                    else:
                        X_copy[col_avg] = pd.Series(0.0, index=X_copy.index, dtype=float)
        
        return X_copy
