import pandas as pd

def load_snp_data(file_path):
    """Load S&P500 data from Excel file."""
    snp_df = pd.read_excel(file_path)
    return snp_df

def load_membership_data(file_path):
    """Load Membership data from Excel file."""
    membership_df = pd.read_excel(file_path)
    return membership_df
