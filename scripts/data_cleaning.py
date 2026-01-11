def clean_snp_data(df):
    """Clean S&P500 data by dropping rows with missing values in key financial columns."""
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Drop rows with missing values in essential columns
    df_cleaned = df.dropna(subset=['gvkey', 'fyear', 'tic'])

    # Drop rows with missing values in specified financial columns
    financial_columns = ['act', 'lct', 'che', 'lt', 'ceq', 'ni', 'sale', 'ebit', 'ebitda',
                         'oancf', 'capx', 'prcc_f', 'csho', 'mkvalt', 'xint']
    df_cleaned = df_cleaned.dropna(subset=financial_columns)

    # Drop duplicates
    df_cleaned = df_cleaned.drop_duplicates()

    # Drop unnecessary columns
    columns_to_drop = ['datadate', 'tstkme', 'bkvlps', 'seq']
    df_cleaned = df_cleaned.drop(columns=columns_to_drop, errors='ignore')

    return df_cleaned

def clean_membership_data(df):
    """Clean Membership data."""
    df.columns = df.columns.str.lower().str.strip()
    df_cleaned = df.dropna(subset=['gvkey', 'fyear'])
    df_cleaned = df_cleaned.drop_duplicates()
    columns_to_drop = ['datadate', 'indfmt', 'consol', 'popsrc', 'datafmt', 'costat', 'curcd']
    df_cleaned = df_cleaned.drop(columns=columns_to_drop, errors='ignore')
    return df_cleaned
