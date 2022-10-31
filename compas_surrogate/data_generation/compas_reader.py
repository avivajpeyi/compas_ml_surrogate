"""Reads COMPAS output data and generates a pandas dataframe"""
import os
import pandas as pd
import h5py as h5  #for handling data format



def read_compas_output(compas_output_path: str ) -> pd.DataFrame:
    """Reads COMPAS output data and generates a pandas dataframe

    Args:
        compas_output_file (str): Name of COMPAS output file

    Returns:
        pd.DataFrame: Pandas dataframe containing COMPAS output data
    """
    # Read COMPAS output file
    fn = os.path.join(compas_output_path, "COMPAS_Output.h5")
    compas_output = h5.File(fn, 'r')

    return compas_output