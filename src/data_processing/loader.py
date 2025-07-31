from pathlib import Path
import pandas as pd
import os

def get_relative_path(absolute_path: Path) -> str:
    """
    Converts an absolute Path object to a string path relative to the project root.
    Assumes the project root is identifiable by 'E-commerce-Bank-Fraud-Detection'.
    """
    path_str = str(absolute_path)
    # Find the index of the project root directory name
    project_root_name = "E-commerce-Bank-Fraud-Detection"
    try:
        index = path_str.rfind(project_root_name)
        if index != -1:
            # Extract the part of the string from the project root onwards
            return path_str[index:]
    except Exception:
        # Fallback if rfind fails or project_root_name is not found
        pass
    return path_str # Return original if relative path cannot be determined


def load_data(file_path: Path, delimiter: str = ',', file_type: str = 'csv', column_dtypes: dict = None) -> pd.DataFrame:
    """
    Loads data from a specified file path into a pandas DataFrame.

    This function supports loading data from CSV and TXT files, handling different
    delimiters. It performs basic checks for file existence and readability.

    Args:
        file_path (Path): The full path to the data file.
        delimiter (str): The delimiter to use for parsing the file (e.g., ',', '|', '\\t').
                         Defaults to ','.
        file_type (str): The type of the file ('csv' or 'txt'). Defaults to 'csv'.
        column_dtypes (dict): A dictionary mapping column names to desired data types
                              (e.g., {'ip_address': str}). This is passed directly to read_csv.

    Returns:
        pd.DataFrame: The loaded DataFrame, or an empty DataFrame if loading fails.
    """
    if not file_path.exists():
        print(f"Error: File not found at {get_relative_path(file_path)}")
        return pd.DataFrame()

    if not file_path.is_file():
        print(f"Error: Path is not a file: {get_relative_path(file_path)}")
        return pd.DataFrame()

    read_kwargs = {'sep': delimiter}
    if column_dtypes:
        read_kwargs['dtype'] = column_dtypes

    try:
        df = pd.DataFrame() # Initialize df to an empty DataFrame
        if file_type == 'csv':
            df = pd.read_csv(file_path, **read_kwargs)
        elif file_type == 'txt':
            df = pd.read_csv(file_path, **read_kwargs)
        else:
            print(f"Error: Unsupported file type '{file_type}'. Only 'csv' and 'txt' are supported.")
            return pd.DataFrame()

        print(f"Successfully loaded {len(df)} rows from {get_relative_path(file_path)}.")
        return df
    except Exception as e:
        print(f"Error loading data from {get_relative_path(file_path)}: {e}")
        return pd.DataFrame() # Return empty pandas DataFrame on error

# Example usage (for independent testing of this script)
if __name__ == "__main__":
    # Define project root for testing purposes
    # Assumes script is in src/data_processing/
    project_root = Path(__file__).resolve().parents[2]
    real_data_file_path = project_root / "data" / "raw" / "creditcard.csv"

    print(f"Attempting to load real data from: {get_relative_path(real_data_file_path)}")
    
    # Test with pandas
    df_cpu = load_data(real_data_file_path, delimiter=',')
    if not df_cpu.empty:
        print("\nDataFrame Head (CPU):")
        print(df_cpu.head())
        print("\nDataFrame Info (CPU):")
        df_cpu.info()
    else:
        print("Failed to load data for CPU example usage.")

    print("\n" + "="*50 + "\n")

    # Test with a non-existent file
    non_existent_file = project_root / "data" / "raw" / "non_existent.csv"
    print(f"Attempting to load non-existent file: {get_relative_path(non_existent_file)}")
    load_data(non_existent_file)

    # Test with a directory path (should fail)
    test_dir = project_root / "data" / "raw"
    print(f"Attempting to load a directory: {get_relative_path(test_dir)}")
    load_data(test_dir)
