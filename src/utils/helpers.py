import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path to allow absolute imports for testing
project_root = Path(__file__).resolve().parents[2] # Adjust based on script's depth
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import get_relative_path for consistent output
from src.data_processing.loader import get_relative_path

def ip_to_int(ip_address_series: pd.Series) -> pd.Series:
    """
    Converts IP addresses from string to integer format.
    Ensures the series is string type first and handles malformed IPs (fewer than 4 parts).
    Handles NaN values by returning NaN for those entries.
    """
    if ip_address_series.empty:
        print("Warning: Input IP address series is empty. Returning empty series.")
        return pd.Series([], dtype='float64') # Return float dtype for consistency

    # Convert to pandas Series and then to string type, coercing errors to NaN
    ip_address_series_pd = ip_address_series.astype(str).replace('nan', np.nan)

    # Handle NaNs: if an IP is NaN, its integer conversion should also be NaN
    valid_ips = ip_address_series_pd.dropna()
    
    if valid_ips.empty:
        return pd.Series(np.nan, index=ip_address_series.index, dtype='float64')

    parts = valid_ips.str.split('.', expand=True)

    # Ensure all 4 parts (0, 1, 2, 3) exist, fill missing with '0' string
    # This handles cases like '192.168.1' -> '192.168.1.0'
    for i in range(4):
        if i not in parts.columns:
            parts[i] = '0'
    # Sort columns to ensure consistent order for arithmetic
    parts = parts[[0, 1, 2, 3]]

    # Convert parts to numeric, coercing errors (e.g., non-numeric parts) to NaN
    # Then fill these NaNs with 0 for calculation, as they are invalid segments
    parts_numeric = parts.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    # Calculate integer representation
    # (A * 256^3) + (B * 256^2) + (C * 256^1) + (D * 256^0)
    ip_int = (
        parts_numeric[0] * (256**3) +
        parts_numeric[1] * (256**2) +
        parts_numeric[2] * (256**1) +
        parts_numeric[3] * (256**0)
    )
    
    # Reindex ip_int to match the original series index, filling NaNs for original NaNs
    result = pd.Series(np.nan, index=ip_address_series.index, dtype='float64')
    result.loc[valid_ips.index] = ip_int

    return result


def merge_ip_to_country(transactions_df: pd.DataFrame, ip_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges transaction data with IP address to country mapping.

    Args:
        transactions_df (pd.DataFrame): DataFrame containing transaction data, including 'ip_address'.
        ip_country_df (pd.DataFrame): DataFrame mapping IP ranges to countries,
                                      with 'lower_bound_ip_address', 'upper_bound_ip_address', 'country'.

    Returns:
        pd.DataFrame: Merged DataFrame with 'country' column added to transactions.
                      Original 'country' column in transactions_df (if any) will be preserved
                      for non-matched IPs, or filled with 'Unknown'.
    """
    print("--- Performing IP Address to Country Merging ---")
    if transactions_df.empty or ip_country_df.empty:
        print("Warning: One or both input DataFrames are empty. Skipping IP-to-Country merge.")
        return transactions_df # Return original transactions_df if empty

    # Ensure IP address columns are strings before conversion
    if 'ip_address' in transactions_df.columns:
        transactions_df['ip_address'] = transactions_df['ip_address'].astype(str)
    else:
        print("Error: 'ip_address' column not found in transactions_df. Cannot perform IP-to-Country merge.")
        return transactions_df

    if 'lower_bound_ip_address' in ip_country_df.columns and 'upper_bound_ip_address' in ip_country_df.columns:
        ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(str)
        ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].astype(str)
    else:
        print("Error: IP range columns ('lower_bound_ip_address', 'upper_bound_ip_address') not found in ip_country_df. Cannot perform IP-to-Country merge.")
        return transactions_df


    print("Converting IP addresses to integer format...")
    # Convert IP addresses to integer for efficient merging
    transactions_df['ip_address_int'] = ip_to_int(transactions_df['ip_address'])
    ip_country_df['lower_bound_ip_address_int'] = ip_to_int(ip_country_df['lower_bound_ip_address'])
    ip_country_df['upper_bound_ip_address_int'] = ip_to_int(ip_country_df['upper_bound_ip_address'])

    # Handle NaNs introduced by ip_to_int conversion
    if transactions_df['ip_address_int'].isnull().any():
        print(f"Warning: {transactions_df['ip_address_int'].isnull().sum()} NaN values found in 'ip_address_int'. These IPs will not be matched.")
    if ip_country_df['lower_bound_ip_address_int'].isnull().any() or ip_country_df['upper_bound_ip_address_int'].isnull().any():
        print(f"Warning: NaN values found in IP range integer columns. Corresponding ranges will not be used for matching.")
        # Drop rows with NaNs in the bounds for the merge, as they are unmatchable
        ip_country_df.dropna(subset=['lower_bound_ip_address_int', 'upper_bound_ip_address_int'], inplace=True)


    print("Matching IP addresses to countries using efficient merge...")
    # Sort for merge_asof
    transactions_df_sorted = transactions_df.sort_values(by='ip_address_int').reset_index(drop=False) # Keep original index
    ip_country_df_sorted = ip_country_df.sort_values(by='lower_bound_ip_address_int').reset_index(drop=True)

    # Perform merge_asof. direction='backward' finds the last row in ip_country_df_sorted
    # whose 'lower_bound_ip_address_int' is less than or equal to 'ip_address_int'.
    merged_df = pd.merge_asof(
        transactions_df_sorted,
        ip_country_df_sorted,
        left_on='ip_address_int',
        right_on='lower_bound_ip_address_int',
        direction='backward',
        suffixes=('', '_ip')
    )

    # Initialize 'country' column in transactions_df_sorted if it doesn't exist, or use existing
    # IMPORTANT: Initialize with 'object' dtype to avoid FutureWarning when assigning strings
    if 'country' not in transactions_df_sorted.columns:
        transactions_df_sorted['country'] = 'Unknown' # Default for unmatched
    
    # Create a Series for the new country values, indexed by the original index
    # FIX: Initialize with 'object' dtype to prevent FutureWarning
    new_country_values = pd.Series(np.nan, index=transactions_df_sorted['index'], dtype='object')

    # Apply the condition and assign country values
    valid_matches_mask = (merged_df['ip_address_int'] >= merged_df['lower_bound_ip_address_int']) & \
                         (merged_df['ip_address_int'] <= merged_df['upper_bound_ip_address_int'])
    
    # FIX: Access 'country' directly. 'country_ip' would only exist if transactions_df already had a 'country' column.
    new_country_values.loc[merged_df.loc[valid_matches_mask, 'index']] = merged_df.loc[valid_matches_mask, 'country']

    # Update the 'country' column in the original transactions_df based on its original index
    # Fill any remaining NaNs (for IPs that didn't match any range) with 'Unknown'
    transactions_df['country'] = new_country_values.reindex(transactions_df.index).fillna('Unknown')


    # Drop temporary IP integer columns from the original transactions_df
    transactions_df = transactions_df.drop(columns=['ip_address_int'], errors='ignore')

    print("IP-to-Country merge complete.")
    return transactions_df


# Example Usage (for independent testing of this script)
if __name__ == "__main__":
    # Define data paths
    RAW_DATA_DIR = project_root / "data" / "raw"
    FRAUD_DATA_PATH = RAW_DATA_DIR / "Fraud_Data.csv"
    IP_TO_COUNTRY_PATH = RAW_DATA_DIR / "IpAddress_to_Country.csv"

    print("--- Starting IP Address Helper Functions Examples (CPU-only) ---")

    # Test ip_to_int
    print("\n--- Testing ip_to_int ---")
    test_ips = pd.Series(['192.168.1.1', '10.0.0.255', '255.255.255.255', '0.0.0.0', 'invalid.ip', np.nan, '1.2.3'])
    int_ips = ip_to_int(test_ips)
    print("Original IPs:\n", test_ips)
    print("\nInteger IPs:\n", int_ips)

    # Test merge_ip_to_country
    print("\n--- Testing merge_ip_to_country ---")
    # Create dummy transaction data
    dummy_transactions = pd.DataFrame({
        'TransactionId': [1, 2, 3, 4, 5],
        'ip_address': ['192.168.1.10', '10.0.0.5', '172.16.0.1', '1.1.1.1', 'invalid.ip'],
        'Amount': [100, 200, 50, 300, 150]
    })

    # Create dummy IP to Country data
    dummy_ip_country = pd.DataFrame({
        'lower_bound_ip_address': ['192.168.1.0', '10.0.0.0', '172.16.0.0', '2.0.0.0'],
        'upper_bound_ip_address': ['192.168.1.255', '10.0.0.255', '172.16.0.255', '2.0.0.255'],
        'country': ['Localhost', 'PrivateNet', 'VPN', 'Cloud']
    })

    print("\nDummy Transactions:\n", dummy_transactions)
    print("\nDummy IP to Country:\n", dummy_ip_country)

    merged_transactions = merge_ip_to_country(dummy_transactions.copy(), dummy_ip_country.copy())
    print("\nMerged Transactions with Country:\n", merged_transactions)

    # Test with real data (if available)
    print("\n--- Testing merge_ip_to_country with real data (if available) ---")
    # Load data using the loader from src.data_processing
    from src.data_processing.loader import load_data
    fraud_data_df_raw = load_data(FRAUD_DATA_PATH, column_dtypes={'ip_address': str})
    ip_country_df_raw = load_data(IP_TO_COUNTRY_PATH, column_dtypes={'lower_bound_ip_address': str, 'upper_bound_ip_address': str})

    if not fraud_data_df_raw.empty and not ip_country_df_raw.empty:
        real_merged_df = merge_ip_to_country(fraud_data_df_raw.head(100).copy(), ip_country_df_raw.copy())
        print("\nReal Merged Data Head:\n", real_merged_df.head())
        print("\nReal Merged Data Country Value Counts:\n", real_merged_df['country'].value_counts().head())
    else:
        print("Skipping real data merge test: Raw fraud data or IP country data not loaded.")

    print("\nIP address helper functions examples complete.")
