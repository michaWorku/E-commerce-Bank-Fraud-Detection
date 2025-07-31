from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import sys
import numpy as np # For np.number in select_dtypes


# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
# Subclasses must implement the inspect method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass


# Concrete Strategy for Data Types Inspection
# --------------------------------------------
# This strategy inspects the data types of each column and counts non-null values.
class DataTypesAndNonNullInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        df_processed = df.copy()

        print("\n--- Data Types and Non-null Counts ---")
        if df_processed.empty:
            print("DataFrame is empty. No data types or non-null counts to display.")
            return

        info_df = pd.DataFrame({
            'Dtype': df_processed.dtypes,
            'Non-Null Count': df_processed.count(),
            'Null Count': df_processed.isnull().sum(),
            'Null Percentage': (df_processed.isnull().sum() / len(df_processed)) * 100
        })
        print(info_df)


# Concrete Strategy for Summary Statistics Inspection
# --------------------------------------------------
# This strategy computes and prints descriptive statistics for numerical columns.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints descriptive statistics for numerical columns of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the descriptive statistics to the console.
        """
        df_processed = df.copy()

        print("\n--- Summary Statistics ---")
        if df_processed.empty:
            print("DataFrame is empty. No summary statistics to display.")
            return

        numerical_df = df_processed.select_dtypes(include=np.number)
        if numerical_df.empty:
            print("No numerical features found for summary statistics.")
            return

        print(numerical_df.describe())


# Context Class for Data Inspection
# ---------------------------------
# This class uses a strategy pattern to perform various data inspections.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): An instance of a concrete inspection strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): A new instance of a concrete inspection strategy.
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)


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

    print("--- Basic Data Inspection Examples (Real Data) ---")

    # Load raw data
    fraud_data_df_raw = load_data(FRAUD_DATA_PATH, column_dtypes={'ip_address': str})
    ip_country_df_raw = load_data(IP_TO_COUNTRY_PATH, column_dtypes={'lower_bound_ip_address': str, 'upper_bound_ip_address': str})

    if not fraud_data_df_raw.empty and not ip_country_df_raw.empty:
        # Merge IP addresses to countries
        fraud_df_merged = merge_ip_to_country(fraud_data_df_raw.copy(), ip_country_df_raw.copy())

        # Initialize the Data Inspector with DataTypesAndNonNullInspectionStrategy
        inspector = DataInspector(DataTypesAndNonNullInspectionStrategy())
        inspector.execute_inspection(fraud_df_merged)

        # Change strategy to SummaryStatisticsInspectionStrategy and execute
        inspector.set_strategy(SummaryStatisticsInspectionStrategy())
        inspector.execute_inspection(fraud_df_merged)
    else:
        print("Skipping real data inspection examples: Raw fraud data or IP country data not loaded.")

    print("\nBasic data inspection examples complete.")

    # Test with an empty DataFrame
    print("\n--- Testing with Empty DataFrame ---")
    empty_df = pd.DataFrame()
    inspector = DataInspector(DataTypesAndNonNullInspectionStrategy()) # Re-initialize for empty test
    inspector.execute_inspection(empty_df) # Should print message for empty df
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.execute_inspection(empty_df) # Should print message for empty df
