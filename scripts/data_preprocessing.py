import numpy as np

def preprocess_data(df):
    """Preprocess data for model training."""
    # Separate features and labels
    X = df.drop(columns=['gvkey', 'fyear', 'tic', 'label'])
    y = df['label']

    return X, y
