import pandas as pd
import numpy as np

def extract_features(df):
    """
    Extract features for distress prediction, round ratios to 5 decimal places,
    keep only relevant columns, and add a label column.
    """
    # Extract features
    df['current_ratio'] = (df['act'] / df['lct']).round(5)
    df['debt_to_equity'] = (df['lt'] / df['ceq']).round(5)
    df['debt_ratio'] = (df['lt'] / df['at']).round(5)

    # Calculate interest coverage ratio, handling cases where xint = 0
    df['interest_coverage'] = np.where(
        df['xint'] == 0,
        1000,  # Assign a high value when there is no interest expense
        (df['ebit'] / df['xint']).round(5)
    )

    df['roa'] = (df['ni'] / df['at']).round(5)
    df['roe'] = (df['ni'] / df['ceq']).round(5)
    df['asset_turnover'] = (df['sale'] / df['at']).round(5)
    df['market_to_book'] = (df['mkvalt'] / df['ceq']).round(5)
    df['cash_flow_to_liabilities'] = (df['oancf'] / df['lt']).round(5)

    df['X1'] = (df['act'] - df['lct']) / df['at']
    df['X2'] = df['re'] / df['at']
    df['X3'] = df['ebit'] / df['at']
    df['X4'] = df['mkvalt'] / df['lt']
    df['X5'] = df['sale'] / df['at']
    df['z_score'] = 1.2 * df['X1'] + 1.4 * df['X2'] + 3.3 * df['X3'] + 0.6 * df['X4'] + 1.0 * df['X5']

    # Composite distress signals
    label_z = (df['z_score'] < 1.81)
    label_ic = (df['interest_coverage'] < 2)
    label_lev = ((df['lt'] / df['at']) > 0.8)

    label_cf = (df['oancf'] < 0) & (df['ni'] < 0)

    # Combine signals
    df['label_raw'] = (label_z & label_ic & label_lev & label_cf).astype(int)

    # Make yearly balanced using 5% worst Z-scores
    df['label_pct'] = df.groupby('fyear')['z_score'].transform(lambda x: (x <= x.quantile(0.1))).astype(int)

    # Final label = either raw distress OR percentile distress
    df['label'] = ((df['label_raw'] == 1) | (df['label_pct'] == 1)).astype(int)

    training_columns = [
        'gvkey', 'fyear', 'tic',  # Identifiers
        # Ratios
        'current_ratio', 'debt_to_equity', 'debt_ratio', 'interest_coverage',
        'roa', 'roe', 'asset_turnover', 'market_to_book', 'cash_flow_to_liabilities',
        # Label
        'label'
    ]

    # Drop all other columns
    training_df = df[training_columns].copy()

    return training_df

def main():
    DATA_DIR = '../data'
    MERGED_FILE = 'merged_data.csv'
    FINAL_FILE = 'training_data.csv'

    # Load merged dataset
    merged_df = pd.read_csv(f"{DATA_DIR}/{MERGED_FILE}")

    # Extract features and prepare training dataset
    training_df = extract_features(merged_df)

    # Save final training dataset
    training_df.to_csv(f"{DATA_DIR}/{FINAL_FILE}", index=False)
    print("Training dataset shape:", training_df.shape)
    print("Training dataset saved to:", f"{DATA_DIR}/{FINAL_FILE}")

if __name__ == "__main__":
    main()
