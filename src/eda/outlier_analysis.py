from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # For numerical operations


# Abstract Base Class for Outlier Analysis Strategy
# -------------------------------------------------
# This class defines a common interface for outlier detection strategies.
# Subclasses must implement the analyze method.
class OutlierAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform outlier analysis on a specific numerical feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed for outliers.

        Returns:
        None: This method visualizes and/or prints outlier information.
        """
        pass


# Concrete Strategy for IQR-based Outlier Analysis
# -------------------------------------------------
# This strategy detects outliers in a numerical feature using the Interquartile Range (IQR) method.
class IQRBasedOutlierAnalysis(OutlierAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Detects and visualizes outliers in a numerical feature using the IQR method.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed for outliers.

        Returns:
        None: Displays a box plot and prints outlier counts.
        """
        df_processed = df.copy()

        if feature not in df_processed.columns:
            print(f"Error: Feature '{feature}' not found in DataFrame.")
            return

        # Explicitly check if the feature is numerical
        if not pd.api.types.is_numeric_dtype(df_processed[feature]):
            print(f"Error: Feature '{feature}' is not numerical. Cannot perform outlier analysis.")
            return

        print(f"\n--- Outlier Analysis for Feature: {feature} ---")
        
        # Drop NaNs for IQR calculation and plotting
        series = df_processed[feature].dropna()

        if series.empty:
            print(f"Feature '{feature}' is empty after dropping NaNs. Cannot perform outlier analysis.")
            return

        # Calculate IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        print(f"IQR for {feature}: {IQR:.2f}")
        print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        print(f"Number of outliers detected: {len(outliers)}")
        if len(outliers) > 0:
            print(f"Outlier percentage: {(len(outliers) / len(series)) * 100:.2f}%")
            print(f"Sample outliers:\n{outliers.head()}")

        # Visualize with a box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=series)
        plt.title(f'Box Plot of {feature} for Outlier Detection')
        plt.xlabel(feature)
        plt.show()


# Context Class for Outlier Analysis
# ----------------------------------
# This class uses a strategy pattern to perform various outlier analyses.
class OutlierAnalyzer:
    def __init__(self, strategy: OutlierAnalysisStrategy):
        """
        Initializes the OutlierAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (OutlierAnalysisStrategy): An instance of a concrete analysis strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierAnalysisStrategy):
        """
        Sets a new analysis strategy.

        Parameters:
        strategy (OutlierAnalysisStrategy): A new instance of a concrete analysis strategy.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the outlier analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        feature (str): The name of the feature to be analyzed for outliers.
        """
        if df.empty:
            print("DataFrame is empty. Cannot perform outlier analysis.")
            return
        self._strategy.analyze(df, feature)


# Example usage for independent testing
if __name__ == "__main__":
    # Add project root to sys.path to allow absolute imports for testing
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.data_processing.loader import load_data
    from src.utils.helpers import merge_ip_to_country

    RAW_DATA_DIR = project_root / "data" / "raw"
    FRAUD_DATA_PATH = RAW_DATA_DIR / "Fraud_Data.csv"
    IP_TO_COUNTRY_PATH = RAW_DATA_DIR / "IpAddress_to_Country.csv"

    print("--- Outlier Analysis Examples (Real Data) ---")

    # Load raw data
    fraud_data_df_raw = load_data(FRAUD_DATA_PATH, column_dtypes={'ip_address': str})
    ip_country_df_raw = load_data(IP_TO_COUNTRY_PATH, column_dtypes={'lower_bound_ip_address': str, 'upper_bound_ip_address': str})

    if not fraud_data_df_raw.empty and not ip_country_df_raw.empty:
        # Merge IP addresses to countries
        fraud_df_merged = merge_ip_to_country(fraud_data_df_raw.copy(), ip_country_df_raw.copy())

        # Simulate preprocessing steps to get numerical features in correct dtypes
        if 'purchase_value' in fraud_df_merged.columns:
            fraud_df_merged['Amount'] = pd.to_numeric(fraud_df_merged['purchase_value'], errors='coerce')
            fraud_df_merged['IsRefund'] = (fraud_df_merged['purchase_value'] < 0).astype(int)
            fraud_df_merged['Amount'] = fraud_df_merged['Amount'].abs()
        if 'age' in fraud_df_merged.columns:
            fraud_df_merged['age'] = pd.to_numeric(fraud_df_merged['age'], errors='coerce')
        if 'purchase_time' in fraud_df_merged.columns:
            fraud_df_merged['TransactionStartTime'] = pd.to_datetime(fraud_df_merged['purchase_time'], errors='coerce', utc=True)
            fraud_df_merged['TransactionHour'] = fraud_df_merged['TransactionStartTime'].dt.hour
            fraud_df_merged['TransactionDayOfWeek'] = fraud_df_merged['TransactionStartTime'].dt.dayofweek
            fraud_df_merged['TransactionMonth'] = fraud_df_merged['TransactionStartTime'].dt.month
            fraud_df_merged['TransactionYear'] = fraud_df_merged['TransactionStartTime'].dt.year
        if 'signup_time' in fraud_df_merged.columns and 'TransactionStartTime' in fraud_df_merged.columns:
             fraud_df_merged['signup_time'] = pd.to_datetime(fraud_df_merged['signup_time'], errors='coerce', utc=True)
             time_diff = (fraud_df_merged['TransactionStartTime'] - fraud_df_merged['signup_time']).dt.total_seconds() / (24 * 3600)
             fraud_df_merged['time_since_signup'] = time_diff.fillna(time_diff.median()).apply(lambda x: x if x >= 0 else 0)

        # Analyze outliers in numerical features
        outlier_analyzer = OutlierAnalyzer(IQRBasedOutlierAnalysis())
        numerical_features_for_outlier = [
            'Amount', 'age', 'time_since_signup', 'IsRefund',
            'TransactionHour', 'TransactionDayOfWeek', 'TransactionMonth', 'TransactionYear'
        ]
        for feature in numerical_features_for_outlier:
            if feature in fraud_df_merged.columns:
                outlier_analyzer.execute_analysis(fraud_df_merged, feature)
            else:
                print(f"Skipping outlier analysis for '{feature}': Column not available after simulated preprocessing.")

        # Example: Test with a non-existent column
        outlier_analyzer.execute_analysis(fraud_df_merged, 'NonExistentColumn')

        # Example: Test with a non-numerical column (should be handled by the strategy)
        outlier_analyzer.execute_analysis(fraud_df_merged, 'source') # 'source' is categorical
    else:
        print("Skipping real data outlier analysis examples: Raw fraud data or IP country data not loaded.")

    print("\nOutlier analysis examples complete.")
