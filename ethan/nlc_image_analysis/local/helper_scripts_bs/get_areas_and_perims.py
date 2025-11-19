"""
Created Nov 18 2024
Updated Nov 18 2024
"""
import pandas as pd
import sys
sys.path.append('/Users/emullen98/Desktop')
import common_code as cc


def get_areas_and_perims(csvloc):
    """
    Given a .csv file location, read its area/perimeter content and return it.

    Parameters
    ----------
    csvloc: (String; REQUIRED)
        Location of the .csv file to be read.

    Returns
    -------
    [0] areas - list of areas from the .csv file
    [1] perims - list of perims from the .csv file
    [2] areax - x-values of the CCDF for areas
    [3] areay - y-values of the CCDF for areas
    [4] perimx - x-values of the CCDF for perims
    [5] perimy - y-values of the CCDF for perims
    """
    df = pd.read_csv(csvloc)
    # Remove rows that have nonzero flag value and area less than 15 pixels
    df = df[df['flag'] == float(0)]
    df = df[df['area'] >= float(15)]
    area = df['area']
    perim = df['perim']
    areas = list(area)
    perims = list(perim)
    areax, areay = cc.ccdf(area, method='dahmen')
    perimx, perimy = cc.ccdf(perim, method='dahmen')

    return areas, perims, areax, areay, perimx, perimy
