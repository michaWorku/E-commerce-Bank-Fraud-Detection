import pandas as pd
import numpy as np

# Attempt to import cudf for GPU acceleration
try:
    import cudf
    _CUDF_AVAILABLE = True
    print("cuDF is available in helpers.py.")
except ImportError:
    _CUDF_AVAILABLE = False
    print("cuDF not available in helpers.py. Falling back to pandas.")

def ip_to_int(ip_address_series):
    """
    Converts IP addresses from string to integer format.
    Ensures the series is string type first and handles malformed IPs (fewer than 4 parts).
    Explicitly converts to pandas Series for string operations if cuDF is active.
    """
    # CRITICAL FIX: Explicitly convert to pandas Series for string operations if cuDF is active.
    # This prevents cuDF's implicit __array__ conversion issues and ensures .str accessor works.
    if _CUDF_AVAILABLE and isinstance(ip_address_series, cudf.Series):
        ip_address_series_pd = ip_address_series.to_pandas().astype(str)
    else:
        ip_address_series_pd = ip_address_series.astype(str)

    parts = ip_address_series_pd.str.split('.', expand=True)

    # Ensure all 4 parts (0, 1, 2, 3) exist, fill missing with '0' string
    for i in range(4):
        if i not in parts.columns:
            parts[i] = '0' # Add missing columns as string '0'
    # Sort columns to ensure consistent order for arithmetic
    parts = parts[[0, 1, 2, 3]]
    
    # Convert parts to numeric (float is fine for intermediate), fillna with 0
    # This handles cases where original parts might be non-numeric or NaN from split
    parts = parts.astype(float).fillna(0)
    
    # Perform arithmetic for integer conversion
    ip_int_pd = (parts[0] * 2**24 + parts[1] * 2**16 + parts[2] * 2**8 + parts[3]).astype(int)

    # Convert back to cuDF Series if original input was cuDF
    if _CUDF_AVAILABLE and isinstance(ip_address_series, cudf.Series):
        return cudf.Series(ip_int_pd, index=ip_address_series.index)
    else:
        return ip_int_pd

def merge_ip_to_country(transactions_df, ip_country_df):
    """
    Merges transaction data with IP address to country mapping.
    Supports both pandas and cuDF DataFrames.
    For cuDF, it converts to pandas for the merge_asof operation due to cuDF's
    current limitations with complex range joins, then converts back.
    """
    is_cudf_input = _CUDF_AVAILABLE and isinstance(transactions_df, cudf.DataFrame)
    
    print("\n--- Performing IP Address to Country Merging ---")
    if transactions_df.empty or ip_country_df.empty:
        print("One or both DataFrames are empty. Skipping IP-to-Country merge.")
        return transactions_df

    # --- CRITICAL FIX: Ensure IP columns are strings before integer conversion ---
    # This ensures that .str accessor is always available.
    # These explicit astype(str) calls are still good practice, but the main fix is in ip_to_int now.
    if 'ip_address' in transactions_df.columns:
        transactions_df['ip_address'] = transactions_df['ip_address'].astype(str)
    if 'lower_bound_ip_address' in ip_country_df.columns:
        ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(str)
    if 'upper_bound_ip_address' in ip_country_df.columns:
        ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].astype(str)
    # --- END CRITICAL FIX ---


    # Convert IP ranges to integer for efficient lookup
    print("Converting IP addresses to integer format...")
    # These will return cuDF Series if input was cuDF, but ip_to_int handles internal conversion
    ip_country_df['lower_bound_ip_address_int'] = ip_to_int(ip_country_df['lower_bound_ip_address'])
    ip_country_df['upper_bound_ip_address_int'] = ip_to_int(ip_country_df['upper_bound_ip_address'])
    transactions_df['ip_address_int'] = ip_to_int(transactions_df['ip_address'])

    # Sort IP country data for binary search if using pandas or for efficient merge
    # Ensure sorting is done on the pandas version if converting
    
    # Initialize 'country' column to a default value (e.g., 'Unknown')
    # Use the appropriate DataFrame type
    if is_cudf_input:
        transactions_df['country'] = cudf.Series(['Unknown'] * len(transactions_df), index=transactions_df.index)
    else:
        transactions_df['country'] = 'Unknown'
    
    print("Matching IP addresses to countries using efficient merge...")
    
    if is_cudf_input:
        # Convert to pandas for the merge_asof operation
        print("Converting to pandas for IP merge (cuDF merge_asof for ranges is limited)...")
        transactions_df_pd = transactions_df.to_pandas()
        ip_country_df_pd = ip_country_df.to_pandas() # Ensure this is also pandas

        ip_country_df_sorted_pd = ip_country_df_pd.sort_values(by='lower_bound_ip_address_int')
        transactions_df_pd_sorted = transactions_df_pd.sort_values(by='ip_address_int')

        # Perform merge_asof
        merged_pd = pd.merge_asof(
            transactions_df_pd_sorted,
            ip_country_df_sorted_pd,
            left_on='ip_address_int',
            right_on='lower_bound_ip_address_int',
            direction='backward', # Find the largest lower_bound_ip_address_int <= ip_address_int
            suffixes=('', '_ip')
        )
        
        # Filter to ensure ip_address_int is within the range [lower, upper]
        # Use .loc for safe assignment
        valid_matches = (merged_pd['ip_address_int'] >= merged_pd['lower_bound_ip_address_int']) & \
                        (merged_pd['ip_address_int'] <= merged_pd['upper_bound_ip_address_int'])
        
        # Create a Series of countries from valid matches, indexed by the original transaction_df_pd_sorted index
        # Ensure matched_countries is a Series with the correct index
        matched_countries = merged_pd.loc[valid_matches, 'country_ip']
        
        # Update the 'country' column in the sorted pandas DataFrame
        # Use .loc for direct assignment based on index alignment
        transactions_df_pd_sorted.loc[matched_countries.index, 'country'] = matched_countries
        
        # Convert back to cuDF DataFrame, maintaining original index if possible
        transactions_df_result = cudf.DataFrame.from_pandas(transactions_df_pd_sorted)

    else: # Pandas path
        transactions_df_sorted = transactions_df.sort_values(by='ip_address_int')
        ip_country_df_sorted = ip_country_df.sort_values(by='lower_bound_ip_address_int')
        
        merged_df = pd.merge_asof(
            transactions_df_sorted,
            ip_country_df_sorted,
            left_on='ip_address_int',
            right_on='lower_bound_ip_address_int',
            direction='backward',
            suffixes=('', '_ip')
        )
        
        # Filter to ensure ip_address_int is within the range [lower, upper]
        # Use .loc for direct assignment based on index alignment
        transactions_df_sorted.loc[:, 'country'] = np.where(
            (merged_df['ip_address_int'] >= merged_df['lower_bound_ip_address_int']) &
            (merged_df['ip_address_int'] <= merged_df['upper_bound_ip_address_int']),
            merged_df['country_ip'], # Use 'country_ip' from the merged DataFrame
            transactions_df_sorted['country'] # Keep 'Unknown' or existing if no match
        )
        transactions_df_result = transactions_df_sorted

    # Drop temporary IP integer columns
    transactions_df_result = transactions_df_result.drop(columns=['ip_address_int'], errors='ignore')
    
    print("IP-to-Country merge complete.")
    return transactions_df_result
