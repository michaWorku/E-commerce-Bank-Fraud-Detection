from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # For numerical operations like np.finfo


# Abstract Base Class for Bivariate Analysis Strategy
# ----------------------------------------------------
# This class defines a common interface for bivariate analysis strategies.
# Subclasses must implement the analyze method.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass


# Concrete Strategy for Numerical vs Numerical Analysis
# ------------------------------------------------------
# This strategy analyzes the relationship between two numerical features using scatter plots.
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Analyzes the relationship between two numerical features using a scatter plot
        and calculates their correlation.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature.
        feature2 (str): The name of the second numerical feature.

        Returns:
        None: Displays a scatter plot and prints the correlation coefficient.
        """
        df_processed = df.copy()

        if feature1 not in df_processed.columns or feature2 not in df_processed.columns:
            print(f"Error: One or both features ('{feature1}', '{feature2}') not found in DataFrame.")
            return

        if not pd.api.types.is_numeric_dtype(df_processed[feature1]) or \
           not pd.api.types.is_numeric_dtype(df_processed[feature2]):
            print(f"Error: Both features ('{feature1}', '{feature2}') must be numerical for this analysis.")
            return

        print(f"\n--- Analyzing Numerical vs Numerical: {feature1} vs {feature2} ---")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df_processed.dropna(subset=[feature1, feature2]))
        plt.title(f'Scatter Plot of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

        # Calculate and print correlation
        correlation = df_processed[feature1].corr(df_processed[feature2])
        print(f"Pearson Correlation between {feature1} and {feature2}: {correlation:.4f}")


# Concrete Strategy for Categorical vs Numerical Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between a categorical and a numerical feature.
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Analyzes the relationship between a categorical feature and a numerical feature
        using a box plot or violin plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature.
        feature2 (str): The name of the numerical feature.

        Returns:
        None: Displays a box plot or violin plot.
        """
        df_processed = df.copy()

        if feature1 not in df_processed.columns or feature2 not in df_processed.columns:
            print(f"Error: One or both features ('{feature1}', '{feature2}') not found in DataFrame.")
            return

        # Determine which is categorical and which is numerical
        is_f1_numeric = pd.api.types.is_numeric_dtype(df_processed[feature1])
        is_f2_numeric = pd.api.types.is_numeric_dtype(df_processed[feature2])

        if (is_f1_numeric and is_f2_numeric) or (not is_f1_numeric and not is_f2_numeric):
            print(f"Error: One feature must be categorical and the other numerical for this analysis. Got '{df_processed[feature1].dtype}' and '{df_processed[feature2].dtype}'.")
            return

        categorical_feature = feature1 if not is_f1_numeric else feature2
        numerical_feature = feature2 if not is_f1_numeric else feature1

        # Check if the identified categorical feature is indeed suitable (e.g., not too many unique values)
        if df_processed[categorical_feature].nunique() > 50:
            print(f"Warning: Categorical feature '{categorical_feature}' has too many unique values ({df_processed[categorical_feature].nunique()}). Plot might be unreadable.")
            # Consider binning or skipping for very high cardinality categorical features

        print(f"\n--- Analyzing Categorical vs Numerical: {categorical_feature} vs {numerical_feature} ---")
        plt.figure(figsize=(12, 7))
        sns.boxplot(x=numerical_feature, y=categorical_feature, data=df_processed.dropna(subset=[categorical_feature, numerical_feature]), palette='viridis')
        plt.title(f'Distribution of {numerical_feature} by {categorical_feature}')
        plt.xlabel(numerical_feature)
        plt.ylabel(categorical_feature)
        plt.tight_layout()
        plt.show()


# Concrete Strategy for Categorical vs Categorical Analysis
# ----------------------------------------------------------
# This strategy analyzes the relationship between two categorical features.
class CategoricalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Analyzes the relationship between two categorical features using a stacked bar chart
        or a heatmap of a crosstab.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first categorical feature.
        feature2 (str): The name of the second categorical feature.

        Returns:
        None: Displays a visualization of the relationship.
        """
        df_processed = df.copy()

        if feature1 not in df_processed.columns or feature2 not in df_processed.columns:
            print(f"Error: One or both features ('{feature1}', '{feature2}') not found in DataFrame.")
            return

        if pd.api.types.is_numeric_dtype(df_processed[feature1]) and df_processed[feature1].nunique() > 50 or \
           pd.api.types.is_numeric_dtype(df_processed[feature2]) and df_processed[feature2].nunique() > 50:
            print(f"Error: Both features ('{feature1}', '{feature2}') must be categorical (or numerical with low cardinality) for this analysis. Cannot perform categorical vs categorical analysis on high-cardinality numerical features.")
            return

        print(f"\n--- Analyzing Categorical vs Categorical: {feature1} vs {feature2} ---")
        plt.figure(figsize=(12, 8))
        
        # Create a crosstab (frequency table)
        crosstab = pd.crosstab(df_processed[feature1], df_processed[feature2])
        
        # Plot as a heatmap
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
        plt.title(f'Crosstab of {feature1} vs {feature2}')
        plt.xlabel(feature2)
        plt.ylabel(feature1)
        plt.tight_layout()
        plt.show()


# Context Class for Bivariate Analysis
# ------------------------------------
# This class uses a strategy pattern to perform various bivariate analyses.
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): An instance of a concrete analysis strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): A new instance of a concrete analysis strategy.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the bivariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.
        """
        if df.empty:
            print("DataFrame is empty. Cannot perform bivariate analysis.")
            return
        self._strategy.analyze(df, feature1, feature2)


# Example usage for independent testing (CPU-only for simplicity)
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

    print("--- Bivariate Analysis Examples (Real Data) ---")

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

        # Numerical vs Numerical
        bivariate_analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
        if 'Amount' in fraud_df_merged.columns and 'age' in fraud_df_merged.columns:
            bivariate_analyzer.execute_analysis(fraud_df_merged, 'Amount', 'age')
        else:
            print("Skipping Numerical vs Numerical analysis: 'Amount' or 'age' not available.")

        # Categorical vs Numerical
        bivariate_analyzer.set_strategy(CategoricalVsNumericalAnalysis())
        if 'country' in fraud_df_merged.columns and 'Amount' in fraud_df_merged.columns:
            bivariate_analyzer.execute_analysis(fraud_df_merged, 'country', 'Amount')
        else:
            print("Skipping Categorical vs Numerical analysis: 'country' or 'Amount' not available.")
        
        if 'class' in fraud_df_merged.columns and 'Amount' in fraud_df_merged.columns:
            # Ensure 'class' is treated as categorical for this analysis
            fraud_df_merged['class'] = fraud_df_merged['class'].astype('category')
            bivariate_analyzer.execute_analysis(fraud_df_merged, 'class', 'Amount')
        else:
            print("Skipping Categorical vs Numerical analysis: 'class' or 'Amount' not available.")


        # Categorical vs Categorical
        bivariate_analyzer.set_strategy(CategoricalVsCategoricalAnalysis())
        if 'source' in fraud_df_merged.columns and 'browser' in fraud_df_merged.columns:
            bivariate_analyzer.execute_analysis(fraud_df_merged, 'source', 'browser')
        else:
            print("Skipping Categorical vs Categorical analysis: 'source' or 'browser' not available.")
        
        if 'country' in fraud_df_merged.columns and 'class' in fraud_df_merged.columns:
            # Ensure 'class' is treated as categorical for this analysis
            fraud_df_merged['class'] = fraud_df_merged['class'].astype('category')
            bivariate_analyzer.execute_analysis(fraud_df_merged, 'country', 'class')
        else:
            print("Skipping Categorical vs Categorical analysis: 'country' or 'class' not available.")


        # Test with non-existent columns
        print("\n--- Testing with Non-Existent Columns ---")
        bivariate_analyzer.execute_analysis(fraud_df_merged, 'NonExistentFeature1', 'Amount')

    else:
        print("Skipping real data bivariate analysis examples: Raw fraud data or IP country data not loaded.")

    print("\nBivariate analysis examples complete.")
