import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocesses the raw rent data for model training.

    This function performs one-hot encoding on categorical features,
    scales the numeric features, and selects final features for the model.

    Args:
        df (pd.DataFrame): The raw DataFrame.
    
    Returns:
        tuple: A tuple containing the processed features (X) and the target (y).

    """

    # --- Feature Selection ---
    features = ['BHK', 'Size', 'City', 'Furnishing Status']
    target = 'Rent'
    data = df[features + [target]].copy()

    # --- One-Hot Encoding for Categorical Features ---
    categorical_cols = ['City', 'Furnishing Status']
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # --- Define Final Features (X) and Target (y) ---
    X = data_encoded.drop(target, axis=1)
    y = data_encoded[target]

    # --- Scale Numeric Features ---
    numeric_cols = ['BHK', 'Size']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler