"""
Created Oct 18 2025
Updated Oct 18 2025
"""
import pandas as pd
import numpy as np


def get_data(csvloc=None, min_area=15):
    """
    Load original NLC area & perimeter data from CSVs

    Parameters
    ----------
    csvloc : str
        Path to the CSV file
        Defaults to None
    min_area : int, optional
        Minimum area threshold in pixels
        Defaults to 15 (per Christian's recommendation)

    Returns
    -------
    tuple
        A tuple containing two numpy arrays: (area, perimeter)

    Raises
    ------
    ValueError
        If csvloc is None
    """
    if csvloc is None:
        raise ValueError("csvloc must be provided")

    df = pd.read_csv(csvloc)
    df = df[df['flag'] == float(0)]  # Remove clouds that have nonzero flag value
    df = df[df['area'] >= float(min_area)]  # Remove clouds with area less than min_area pixels
    area = df['area']
    perim = df['perim']

    return np.array(area), np.array(perim)
