from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Attempt to import cudf for GPU acceleration
try:
    import cudf
    _CUDF_AVAILABLE_EDA = True
except ImportError:
    _CUDF_AVAILABLE_EDA = False


# Abstract Base Class for Temporal Analysis Strategy
# --------------------------------------------------
# This class defines a common interface for temporal analysis strategies.
# Subclasses must implement the analyze method.
class TemporalAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, time_column: str, metrics: list):
        """
        Perform temporal analysis on a dataframe, visualizing trends of specified metrics
        over time.

        Parameters:
        df (pd.DataFrame): The dataframe containing the time-series data.
        time_column (str): The name of the datetime column to use for temporal analysis.
        metrics (list): A list of numerical columns to aggregate and plot as trends.

        Returns:
        None: This method visualizes temporal trends.
        """
        pass


# Concrete Strategy for Monthly Trend Analysis
# --------------------------------------------
# This strategy aggregates data monthly and plots trends for specified metrics.
class MonthlyTrendAnalysis(TemporalAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, time_column: str, metrics: list):
        """
        Analyzes and visualizes monthly trends for specified numerical metrics.
        Handles both pandas and cuDF DataFrames by converting to pandas for analysis.

        Parameters:
        df (pd.DataFrame): The dataframe containing the time-series data.
        time_column (str): The name of the datetime column to use for temporal analysis.
        metrics (list): A list of numerical columns to aggregate and plot as trends.

        Returns:
        None: Displays line plots of monthly trends.
        """
        # Convert to pandas if input is cuDF
        if _CUDF_AVAILABLE_EDA and isinstance(df, cudf.DataFrame):
            df_processed = df.to_pandas()
        else:
            df_processed = df.copy()

        if time_column not in df_processed.columns:
            print(f"Error: Time column '{time_column}' not found in DataFrame. Cannot perform temporal analysis.")
            return

        # Ensure time_column is datetime
        df_processed[time_column] = pd.to_datetime(df_processed[time_column], errors='coerce', utc=True)
        df_processed.dropna(subset=[time_column], inplace=True)

        if df_processed.empty:
            print("DataFrame is empty after time column processing. Cannot perform temporal analysis.")
            return

        print(f"\n--- Analyzing Monthly Trends for {', '.join(metrics)} ---")

        # Extract YearMonth for aggregation
        # UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
        # This warning is expected and acceptable for EDA purposes.
        df_processed['YearMonth'] = df_processed[time_column].dt.to_period('M')

        valid_metrics = []
        for metric in metrics:
            if metric not in df_processed.columns:
                print(f"Warning: Metric '{metric}' not found. Skipping.")
                continue
            if not pd.api.types.is_numeric_dtype(df_processed[metric]):
                print(f"Warning: Metric '{metric}' is not numerical. Skipping.")
                continue
            valid_metrics.append(metric)

        if not valid_metrics:
            print("No valid numerical metrics found for temporal analysis.")
            return

        # Aggregate data by YearMonth
        monthly_trends = df_processed.groupby('YearMonth')[valid_metrics].mean().reset_index()
        # Convert YearMonth back to datetime for plotting for better x-axis formatting
        monthly_trends['YearMonth'] = monthly_trends['YearMonth'].dt.to_timestamp()

        for metric in valid_metrics:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='YearMonth', y=metric, data=monthly_trends, marker='o')
            plt.title(f'Monthly Trend of {metric}')
            plt.xlabel('Date')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


# Context Class for Temporal Analysis
# -----------------------------------
# This class uses a strategy pattern to perform various temporal analyses.
class TemporalAnalyzer:
    def __init__(self, strategy: TemporalAnalysisStrategy):
        """
        Initializes the TemporalAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (TemporalAnalysisStrategy): An instance of a concrete analysis strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: TemporalAnalysisStrategy):
        """
        Sets a new analysis strategy.

        Parameters:
        strategy (TemporalAnalysisStrategy): A new instance of a concrete analysis strategy.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, time_column: str, metrics: list):
        """
        Executes the temporal analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        time_column (str): The name of the datetime column.
        metrics (list): A list of numerical columns to aggregate and plot.
        """
        if df.empty:
            print("DataFrame is empty. Cannot perform temporal analysis.")
            return
        self._strategy.analyze(df, time_column, metrics)


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

    print("--- Temporal Analysis Examples (Real Data) ---")

    # Load raw data (using pandas for independent testing simplicity)
    fraud_data_df_raw = load_data(FRAUD_DATA_PATH, use_gpu=False, column_dtypes={'ip_address': str})
    ip_country_df_raw = load_data(IP_TO_COUNTRY_PATH, use_gpu=False, column_dtypes={'lower_bound_ip_address': str, 'upper_bound_ip_address': str})

    if not fraud_data_df_raw.empty and not ip_country_df_raw.empty:
        # Merge IP addresses to countries
        fraud_df_merged = merge_ip_to_country(fraud_data_df_raw.copy(), ip_country_df_raw.copy())

        # Simulate preprocessing steps to get numerical and datetime features in correct dtypes
        if 'purchase_value' in fraud_df_merged.columns:
            fraud_df_merged['Amount'] = pd.to_numeric(fraud_df_merged['purchase_value'], errors='coerce')
        if 'purchase_time' in fraud_df_merged.columns:
            fraud_df_merged['TransactionStartTime'] = pd.to_datetime(fraud_df_merged['purchase_time'], errors='coerce', utc=True)
            
        # Example 1: Analyze monthly trends for 'Amount'
        temporal_analyzer = TemporalAnalyzer(MonthlyTrendAnalysis())
        if 'TransactionStartTime' in fraud_df_merged.columns and 'Amount' in fraud_df_merged.columns:
            temporal_analyzer.execute_analysis(fraud_df_merged, 'TransactionStartTime', ['Amount'])
        else:
            print("Skipping temporal analysis: 'TransactionStartTime' or 'Amount' not available after simulated preprocessing.")

        # Example 2: Test with a non-existent time column
        print("\n--- Example 2: Testing with Non-Existent Time Column ---")
        temporal_analyzer.execute_analysis(fraud_df_merged, 'NonExistentTimeColumn', ['Amount'])

        # Example 3: Test with a non-numerical metric (should be skipped or warned by the strategy)
        print("\n--- Example 3: Testing with Non-Numerical Metric ---")
        temporal_analyzer.execute_analysis(fraud_df_merged, 'TransactionStartTime', ['source']) # 'source' is categorical
    else:
        print("Skipping real data temporal analysis examples: Raw fraud data or IP country data not loaded.")

    print("\nTemporal analysis examples complete.")
