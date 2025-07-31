from abc import ABC, abstractmethod
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np # For np.nan


# Abstract Base Class for Missing Values Analysis
# -----------------------------------------------
# This class defines a template for missing values analysis.
# Subclasses must implement the methods to identify and visualize missing values.
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        df_processed = df.copy()

        if df_processed.empty:
            print("DataFrame is empty. Cannot perform missing values analysis.")
            return
        self.identify_missing_values(df_processed)
        self.visualize_missing_values(df_processed)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method prints the missing values information.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: This method displays a visualization.
        """
        pass


# Concrete Implementation for Simple Missing Values Analysis
# -----------------------------------------------------------
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies and prints the count and percentage of missing values per column.
        """
        print("\n--- Missing Values Count by Column ---")
        missing_counts = df.isnull().sum()
        missing_percentage = (df.isnull().sum() / len(df)) * 100

        missing_info = pd.DataFrame({
            'Missing Count': missing_counts,
            'Missing Percentage': missing_percentage
        })
        # Filter to show only columns with missing values
        missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

        if missing_info.empty:
            print("No missing values found in the DataFrame.")
        else:
            print(missing_info)


    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap to visualize the missing values in the dataframe.
        """
        print("\nVisualizing Missing Values...")
        if df.isnull().sum().sum() == 0:
            print("No missing values to visualize.")
            return

        plt.figure(figsize=(12, 8))
        # Use a different cmap for better visibility if data is sparse
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()


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
    CREDITCARD_DATA_PATH = RAW_DATA_DIR / "creditcard.csv" # Also load credit card for its missing values

    print("--- Missing Values Analysis Examples (Real Data) ---")

    # Example 1: Fraud Data
    print("\n--- Missing Values Analysis for E-commerce Fraud Data ---")
    fraud_data_df_raw = load_data(FRAUD_DATA_PATH, column_dtypes={'ip_address': str})
    ip_country_df_raw = load_data(IP_TO_COUNTRY_PATH, column_dtypes={'lower_bound_ip_address': str, 'upper_bound_ip_address': str})

    if not fraud_data_df_raw.empty and not ip_country_df_raw.empty:
        fraud_df_merged = merge_ip_to_country(fraud_data_df_raw.copy(), ip_country_df_raw.copy())
        # Simulate some preprocessing to introduce potential NaNs or ensure correct dtypes
        if 'purchase_value' in fraud_df_merged.columns:
            fraud_df_merged['Amount'] = pd.to_numeric(fraud_df_merged['purchase_value'], errors='coerce')
        if 'age' in fraud_df_merged.columns:
            fraud_df_merged['age'] = pd.to_numeric(fraud_df_merged['age'], errors='coerce')
        
        missing_values_analyzer = SimpleMissingValuesAnalysis()
        missing_values_analyzer.analyze(fraud_df_merged)
    else:
        print("Skipping Fraud Data Missing Values Analysis: Raw fraud data or IP country data not loaded.")

    # Example 2: Credit Card Data (known to have missing values in some contexts)
    print("\n--- Missing Values Analysis for Bank Credit Card Fraud Data ---")
    creditcard_df_raw = load_data(CREDITCARD_DATA_PATH)

    if not creditcard_df_raw.empty:
        # Simulate preprocessing steps that might introduce NaNs if they are not already present
        if 'Time' in creditcard_df_raw.columns:
            creditcard_df_raw['Time'] = pd.to_numeric(creditcard_df_raw['Time'], errors='coerce')
        if 'Amount' in creditcard_df_raw.columns:
            creditcard_df_raw['Amount'] = pd.to_numeric(creditcard_df_raw['Amount'], errors='coerce')
            creditcard_df_raw['IsRefund'] = (creditcard_df_raw['Amount'] < 0).astype(int) # Add IsRefund for consistency

        missing_values_analyzer = SimpleMissingValuesAnalysis()
        missing_values_analyzer.analyze(creditcard_df_raw)
    else:
        print("Skipping Credit Card Data Missing Values Analysis: Raw credit card data not loaded.")

    print("\nMissing values analysis examples complete.")
