import pandas as pd

def clean_rent_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the house rent dataset by:
    - Removing outliers in Rent (> 1,00,000)
    - Dropping irrelevant columns
    - Returning the cleaned DataFrame
    """
    
    # 1. Remove outliers in Rent
    df = df[df['Rent'] <= 100000]
    
    # 2. Drop non-informative columns
    df.drop(columns=['Area Type', 'Point of Contact', 'Posted On'], inplace=True)
    
    # 3. Reset index
    df.reset_index(drop=True, inplace=True)
    
    return df

