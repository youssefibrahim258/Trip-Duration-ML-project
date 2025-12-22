import pandas as pd 


def load_data(file_path:str) ->pd.DataFrame:
    """
        Load CSV Data into a DataFrame

    Args:
        file_path (str): Path to the CSV File

    Returns:
        pd.DataFrame: Loaded Dataset
    """

    return pd.read_csv(file_path)


    