#%%
from typing import Dict, Union
import random

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
#%%
def tab2text(
    missing: pd.DataFrame, 
    idx: Union[int, slice, str],
    permutation=False) -> str:
    """
    Convert a row from a DataFrame into a text string.
    
    This function selects a single instance from the tabular data,
    optionally shuffles the order of the key-value pairs, and converts it into
    a string of the format "column_name is column_value".
    
    Parameters:
        missing (pd.DataFrame): The DataFrame containing the data.
        idx (int, slice, or str): The index or label for the row to be converted.
        permutation (bool): If True, shuffle the order of the key-value pairs.
    
    Returns:
        str: A text string representing the row.
    """

    if isinstance(idx, int):
        row = missing.iloc[idx]
    else:
        row = missing.loc[idx]
    

    key = list(range(len(row)))

    if permutation:
        random.shuffle(key) 

    text = ", ".join(
        [
            "%s is %s" % (row.index[i].replace(",", ""), str(row[i]).strip().replace(",", ""))
            for i in key
        ]
    )

    return text
#%%
def tab2json(
    missing: pd.DataFrame, 
    idx: Union[int, slice, str],
    permutation: bool = False
) -> Union[dict, list]:
    """
    Convert a row from a DataFrame into a dictionary with the format {column_name: value}.
    
    This function selects a single instance from the tabular data and converts it
    into a dictionary where keys are column names and values are the corresponding data.
    If permutation is True, the order of the key-value pairs will be randomly shuffled 
    
    Parameters:
        missing (pd.DataFrame): The DataFrame containing the data.
        idx (int, slice, or str): The index or label for the row to be converted.
        permutation (bool): If True, shuffle the order of the key-value pairs.
    
    Returns:
        dict: A dictionary representing the row in {column_name: value} format.
    """
    # Select the observation(s)
    if isinstance(idx, int) or isinstance(missing.loc[idx], pd.Series):
        # Single observation selected as Series
        row = missing.iloc[idx] if isinstance(idx, int) else missing.loc[idx]
        data_dict = row.to_dict()
        if permutation:
            items = list(data_dict.items())
            random.shuffle(items)
            data_dict = dict(items)
            
    return data_dict
#%%
def tab2num(
    missing: pd.DataFrame, 
    idx: Union[int, slice, str],
    permutation: bool = False
) -> np.ndarray:
    """
    Convert a row from a DataFrame into a numerical embedding vector.
    
    This function selects a single instance from the tabular data and attempts to
    convert each value to a numerical format.
    
    Parameters:
        missing (pd.DataFrame): The DataFrame containing the data.
        idx (int, slice, or str): The index or label for the row to be converted.
    
    Returns:
        np.ndarray: A numerical vector representing the row.
    """
    if isinstance(idx, int):
        row = missing.iloc[idx]
    else:
        row = missing.loc[idx]
    
    # Convert row values to strings
    values = list(row.astype(str).values)
    
    if permutation:
        random.shuffle(values)
    
    # Join only the values into a comma-separated string
    text = ", ".join(values)
    
    return text
#%%