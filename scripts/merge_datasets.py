import pandas as pd
from data_loading import load_snp_data, load_membership_data
from data_cleaning import clean_snp_data, clean_membership_data


def select_top_800_per_year(df):
    df_sorted = df.sort_values(['fyear', 'mkvalt'], ascending=[True, False])

    top_500_per_year = (
        df_sorted.groupby('fyear')
        .head(800)
        .reset_index(drop=True)
    )

    return top_500_per_year

def main():
    DATA_DIR = '../data'
    SNP_FILE = 'Data_S&P500.xlsx'
    MEMBERSHIP_FILE = 'Data_membership.xlsx'
    MERGED_FILE = 'merged_data.csv'

    # Load data
    snp_df = load_snp_data(f"{DATA_DIR}/{SNP_FILE}")
    membership_df = load_membership_data(f"{DATA_DIR}/{MEMBERSHIP_FILE}")

    # Clean data
    snp_df_cleaned = clean_snp_data(snp_df)
    membership_df_cleaned = clean_membership_data(membership_df)

    # Merge datasets
    merged_df = pd.merge(
        snp_df_cleaned,
        membership_df_cleaned,
        on=['gvkey', 'fyear'],
        how='inner'
    )

    # Create the filtered DataFrame
    top_500_df = select_top_500_per_year(merged_df)
    top_500_df = top_500_df.drop_duplicates(subset=['fyear', 'tic'], keep='first')

    # Save final dataset
    top_500_df.to_csv(f"{DATA_DIR}/{MERGED_FILE}", index=False)
    print("Merged dataset shape:", top_500_df.shape)

if __name__ == "__main__":
    main()
