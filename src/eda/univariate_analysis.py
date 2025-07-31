from abc import ABC, abstractmethod
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np # Import numpy for numerical type checking


# Abstract Base Class for Univariate Analysis Strategy
# -----------------------------------------------------
# This class defines a common interface for univariate analysis strategies.
# Subclasses must implement the analyze method.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass


# Concrete Strategy for Numerical Features
# -----------------------------------------
# This strategy analyzes numerical features by plotting their distribution.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using a histogram and a box plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a histogram and box plot of the feature.
        """
        df_processed = df.copy()

        if feature not in df_processed.columns:
            print(f"Error: Feature '{feature}' not found in DataFrame.")
            return

        # Explicitly check if the feature is numerical
        if not pd.api.types.is_numeric_dtype(df_processed[feature]):
            print(f"Error: Feature '{feature}' is not numerical. Cannot perform numerical univariate analysis.")
            return

        print(f"\n--- Analyzing Numerical Feature: {feature} ---")
        plt.figure(figsize=(10, 6))

        # Histogram
        plt.subplot(2, 1, 1) # 2 rows, 1 column, 1st plot
        sns.histplot(df_processed[feature].dropna(), kde=True, bins=50)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

        # Box plot for outliers
        plt.subplot(2, 1, 2) # 2 rows, 1 column, 2nd plot
        sns.boxplot(x=df_processed[feature].dropna())
        plt.title(f'Box Plot of {feature}')
        plt.xlabel(feature)

        plt.tight_layout()
        plt.show()


# Concrete Strategy for Categorical Features
# -------------------------------------------
# This strategy analyzes categorical features by plotting their value counts.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature using a count plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a count plot of the feature.
        """
        df_processed = df.copy()

        if feature not in df_processed.columns:
            print(f"Error: Feature '{feature}' not found in DataFrame.")
            return

        # Explicitly check if the feature is categorical (not numerical)
        # We allow object, category, bool, and even numerical if it has few unique values (e.g., encoded categories)
        if pd.api.types.is_numeric_dtype(df_processed[feature]) and df_processed[feature].nunique() > 50:
             print(f"Warning: Feature '{feature}' is numerical with many unique values. Consider numerical analysis or binning for categorical view.")
             # Still proceed if user explicitly asked for categorical analysis
        
        print(f"\n--- Analyzing Categorical Feature: {feature} ---")
        plt.figure(figsize=(10, 6))
        sns.countplot(y=feature, data=df_processed, order=df_processed[feature].value_counts().index, palette='viridis')
        plt.title(f'Distribution of {feature}')
        plt.xlabel('Count')
        plt.ylabel(feature)
        plt.tight_layout()
        plt.show()


# Context Class for Univariate Analysis
# -------------------------------------
# This class uses a strategy pattern to perform various univariate analyses.
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): An instance of a concrete analysis strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): A new instance of a concrete analysis strategy.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        feature (str): The name of the feature/column to be analyzed.
        """
        if df.empty:
            print("DataFrame is empty. Cannot perform univariate analysis.")
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

    print("--- Univariate Analysis Examples (Real Data) ---")

    # Load raw data
    fraud_data_df_raw = load_data(FRAUD_DATA_PATH, column_dtypes={'ip_address': str})
    ip_country_df_raw = load_data(IP_TO_COUNTRY_PATH, column_dtypes={'lower_bound_ip_address': str, 'upper_bound_ip_address': str})

    if not fraud_data_df_raw.empty and not ip_country_df_raw.empty:
        # Merge IP addresses to countries
        fraud_df_merged = merge_ip_to_country(fraud_data_df_raw.copy(), ip_country_df_raw.copy())

        # Simulate preprocessing steps to get numerical features in correct dtypes
        # This is a simplified version, in a real scenario you'd load processed_df
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


        # Numerical feature analysis
        numerical_features_for_eda = [
            'Amount', 'age', 'IsRefund', 'TransactionHour', 'TransactionDayOfWeek',
            'TransactionMonth', 'TransactionYear', 'time_since_signup'
        ]
        analyzer_num = UnivariateAnalyzer(NumericalUnivariateAnalysis())
        for feature in numerical_features_for_eda:
            if feature in fraud_df_merged.columns:
                analyzer_num.execute_analysis(fraud_df_merged, feature)
            else:
                print(f"Skipping numerical analysis for '{feature}': Column not available after simulated preprocessing.")

        # Categorical feature analysis
        categorical_features_for_eda = ['source', 'browser', 'sex', 'country', 'class']
        analyzer_cat = UnivariateAnalyzer(CategoricalUnivariateAnalysis())
        for feature in categorical_features_for_eda:
            if feature in fraud_df_merged.columns:
                analyzer_cat.execute_analysis(fraud_df_merged, feature)
            else:
                print(f"Skipping categorical analysis for '{feature}': Column not available.")

        # Test with non-existent feature
        analyzer_num.execute_analysis(fraud_df_merged, 'NonExistentFeature')

    else:
        print("Skipping real data univariate analysis examples: Raw fraud data or IP country data not loaded.")

    print("\nUnivariate analysis examples complete.")
