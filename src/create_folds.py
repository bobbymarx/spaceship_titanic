import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import config

def create_folds(data, num_folds=8):
    """
    Creates stratified k-folds and adds a 'kfold' column to the dataframe
    
    Parameters:
    -----------
    data: pandas DataFrame
        Input training data
    num_folds: int
        Number of folds to create (default: 5)
    
    Returns:
    --------
    pandas DataFrame with added kfold column
    """
    # Create a new column for kfold
    data["kfold"] = -1
    
    # Randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Assuming the target column is named 'Transported'
    y = data.Transported.values
    
    # Initialize the kfold class
    skf = StratifiedKFold(n_splits=num_folds)
    
    # Fill in the kfold column
    for fold, (_, valid_idx) in enumerate(skf.split(X=data, y=y)):
        data.loc[valid_idx, "kfold"] = fold
    
    return data

if __name__ == "__main__":
    # Read the training data from input folder
    df = pd.read_csv(config.Training_File)
    
    # Create folds
    df_with_folds = create_folds(df)
    
    # Create output directory if it doesn't exist
    os.makedirs("input", exist_ok=True)
    
    # Save the new CSV with folds to the 'input' folder
    df_with_folds.to_csv("input/train_folds.csv", index=False)
    print(f"Created 8 folds and saved to input/train_folds.csv")