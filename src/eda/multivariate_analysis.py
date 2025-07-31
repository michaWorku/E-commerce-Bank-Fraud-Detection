from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Abstract Base Class for Multivariate Analysis
# ----------------------------------------------
# This class defines a template for performing multivariate analysis.
# Subclasses can override specific steps like correlation heatmap and pair plot generation.
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame, features: list = None):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.
        features (list, optional): A list of features to include in the pair plot.
                                   If None, all numerical features will be used.

        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        df_processed = df.copy()

        if df_processed.empty:
            print("DataFrame is empty. Cannot perform multivariate analysis.")
            return

        print("\n--- Performing Multivariate Analysis ---")

        # Select numerical features for correlation heatmap
        numerical_df = df_processed.select_dtypes(include=np.number)
        if numerical_df.empty:
            print("No numerical features found for correlation heatmap.")
        else:
            self.generate_correlation_heatmap(numerical_df)

        # Select features for pair plot
        plot_features = features if features is not None else numerical_df.columns.tolist()
        
        # Filter plot_features to only include existing numerical columns
        plot_features_existing_numerical = [col for col in plot_features if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]

        if not plot_features_existing_numerical:
            print("No valid numerical features found for pair plot.")
        else:
            self.generate_pair_plot(df_processed[plot_features_existing_numerical])


    @abstractmethod
    def generate_correlation_heatmap(self, df_numerical: pd.DataFrame):
        """
        Generates a correlation heatmap for numerical features.

        Parameters:
        df_numerical (pd.DataFrame): DataFrame containing only numerical features.
        """
        pass

    @abstractmethod
    def generate_pair_plot(self, df_numerical: pd.DataFrame):
        """
        Generates a pair plot for selected numerical features.

        Parameters:
        df_numerical (pd.DataFrame): DataFrame containing selected numerical features for the pair plot.
        """
        pass


# Concrete Implementation of Multivariate Analysis
# -------------------------------------------------
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df_numerical: pd.DataFrame):
        """
        Generates a correlation heatmap.
        """
        print("\nGenerating Correlation Heatmap...")
        if df_numerical.empty:
            print("Numerical DataFrame is empty. Cannot generate correlation heatmap.")
            return
        
        # Calculate correlation matrix, dropping rows with any NaN for correlation calculation
        corr_matrix = df_numerical.corr(numeric_only=True) # Ensure only numeric columns are considered

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.show()

    def generate_pair_plot(self, plot_df: pd.DataFrame):
        """
        Generates a pair plot for selected numerical features.
        """
        print("\nGenerating Pair Plot...")
        if plot_df.empty:
            print("DataFrame for pair plot is empty. Cannot generate pair plot.")
            return

        # Drop NaNs for pairplot to avoid errors, only from the columns being plotted
        # Ensure that the columns are numerical before passing to pairplot
        numeric_plot_df = plot_df.select_dtypes(include=np.number)
        
        if numeric_plot_df.empty:
            print("No numerical features found in the selected columns for pair plot.")
            return

        sns.pairplot(numeric_plot_df.dropna())
        plt.suptitle('Pair Plot of Selected Numerical Features', y=1.02) # Adjust title position
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

    print("--- Multivariate Analysis Examples (Real Data) ---")

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

        # Example 1: Correlation Heatmap (using all relevant numerical columns)
        multivariate_analyzer = SimpleMultivariateAnalysis()
        
        # Define numerical features for multivariate analysis
        numerical_features_for_multivariate = [
            'Amount', 'age', 'IsRefund', 'TransactionHour', 'TransactionDayOfWeek',
            'TransactionMonth', 'TransactionYear', 'time_since_signup'
        ]
        # Filter to only include columns that exist and are numerical after simulated preprocessing
        existing_numerical_features = [col for col in numerical_features_for_multivariate if col in fraud_df_merged.columns and pd.api.types.is_numeric_dtype(fraud_df_merged[col])]

        if existing_numerical_features:
            multivariate_analyzer.analyze(fraud_df_merged[existing_numerical_features])
        else:
            print("No valid numerical features found for multivariate analysis after simulated preprocessing.")


        # Example 2: Pair Plot of selected important numerical features
        selected_features_for_pairplot = ['Amount', 'age'] # Using 'Amount' and 'age' from fraud data
        # Filter to only include columns that exist and are numerical
        existing_selected_features = [col for col in selected_features_for_pairplot if col in fraud_df_merged.columns and pd.api.types.is_numeric_dtype(fraud_df_merged[col])]

        if existing_selected_features:
            multivariate_analyzer.analyze(fraud_df_merged, features=existing_selected_features)
        else:
            print("No valid selected numerical features found for pair plot after simulated preprocessing.")


        # Test with non-numerical features in the 'features' list (should be filtered internally)
        print("\n--- Testing with mixed types in features list (should filter out non-numerical) ---")
        multivariate_analyzer.analyze(fraud_df_merged, features=['Amount', 'source', 'age']) # 'source' is categorical

    else:
        print("Skipping real data multivariate analysis examples: Raw fraud data or IP country data not loaded.")

    print("\nMultivariate analysis examples complete.")
