from pathlib import Path
import pandas as pd

# Attempt to import cudf for GPU acceleration
try:
    import cudf
    _CUDF_AVAILABLE = True
    print("cuDF is available. Data loading can be accelerated on GPU.")
except ImportError:
    _CUDF_AVAILABLE = False
    print("cuDF not available. Falling back to pandas for data loading (CPU).")

def load_data(file_path: Path, delimiter: str = ',', file_type: str = 'csv', use_gpu: bool = False, column_dtypes: dict = None) -> pd.DataFrame:
    """
    Loads data from a specified file path into a pandas or cuDF DataFrame.

    This function supports loading data from CSV and TXT files, handling different
    delimiters. It performs basic checks for file existence and readability.

    Args:
        file_path (Path): The full path to the data file.
        delimiter (str): The delimiter to use for parsing the file (e.g., ',', '|', '\t').
                         Defaults to ','.
        file_type (str): The type of the file ('csv' or 'txt'). Defaults to 'csv'.
        use_gpu (bool): If True and cuDF is available, data will be loaded into a cuDF DataFrame.
                        Otherwise, a pandas DataFrame is used. Defaults to False.
        column_dtypes (dict): A dictionary mapping column names to desired data types
                              (e.g., {'ip_address': str}). This is passed directly to read_csv.

    Returns:
        pd.DataFrame or cudf.DataFrame: The loaded DataFrame, or an empty DataFrame if loading fails.
    """
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame() # Always return pandas DataFrame if empty

    try:
        read_csv_kwargs = {'delimiter': delimiter}
        if column_dtypes:
            read_csv_kwargs['dtype'] = column_dtypes

        if use_gpu and _CUDF_AVAILABLE:
            print(f"Attempting to load data from {file_path} using cuDF (GPU)...")
            if file_type == 'csv':
                df = cudf.read_csv(file_path, **read_csv_kwargs)
            elif file_type == 'txt':
                df = cudf.read_csv(file_path, **read_csv_kwargs)
            else:
                print(f"Error: Unsupported file type '{file_type}' for cuDF. Only 'csv' and 'txt' are supported.")
                return pd.DataFrame()
        else:
            print(f"Loading data from {file_path} using pandas (CPU)...")
            if file_type == 'csv':
                df = pd.read_csv(file_path, **read_csv_kwargs)
            elif file_type == 'txt':
                df = pd.read_csv(file_path, **read_csv_kwargs)
            else:
                print(f"Error: Unsupported file type '{file_type}'. Only 'csv' and 'txt' are supported.")
                return pd.DataFrame()

        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame() # Return empty pandas DataFrame on error

# Example usage (for independent testing of this script)
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    real_data_file_path = project_root / "data" / "raw" / "creditcard.csv"

    print(f"Attempting to load real data from: {real_data_file_path}")
    
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

    # Test with cuDF if available
    if _CUDF_AVAILABLE:
        df_gpu = load_data(real_data_file_path, delimiter=',', use_gpu=True)
        if not df_gpu.empty:
            print("\nDataFrame Head (GPU):")
            print(df_gpu.head())
            print("\nDataFrame Info (GPU):")
            df_gpu.info()
        else:
            print("Failed to load data for GPU example usage.")
    else:
        print("Skipping GPU test: cuDF not available.")
