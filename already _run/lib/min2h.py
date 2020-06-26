"""
This Skrips takes pandas dataframe
and sums each 4 rows togther
Intended for use in the Project "Stromdaten-Analyse" for DatascienceI SS2020
at the Goethe University.
"""
import pandas as pd
from os import getcwd


def min_to_h(my_data):
    """Input Dataframe to sum 4x 15 min up to 1h"""
    my_new_data = my_data.iloc[: len(my_data)//4 + 1 , :].copy()  # We create a new data Frame to save our data to conserve the Data frame structure

    for i in range(0, len(my_data), 4):
        my_new_data.iloc[i // 4, 2:] = my_data.iloc[i, 2:] + my_data.iloc[i + 1, 2:] + my_data.iloc[i + 2, 2:] + my_data.iloc[i + 3, 2:]
        my_new_data.iloc[i // 4, :2] = my_data.iloc[i, :2]

    return(my_new_data)