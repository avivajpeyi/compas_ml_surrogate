"""Module to load data from HDF5 files"""
import inspect
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd


def parse_h5_file(file_path: str) -> Dict[str, Any]:
    """
    Parse an HDF5 file and return a dictionary containing the contents

    :param file_path: str
        Path to the HDF5 file

    :return: dict
        Dictionary containing the contents of the HDF5 file
    """
    with h5py.File(file_path, "r") as h5_file:
        return recursively_load_dict_contents_from_group(h5_file, "/")


def recursively_load_dict_contents_from_group(
    h5_file: h5py.File, path: str
) -> Dict[str, Any]:
    """
    Recursively load a HDF5 file into a dictionary

    :param h5_file: the h5py file object
    :param path: the path to the group in the hdf5 file
    :return: a dictionary containing the contents of the hdf5 file
    """

    output = dict()
    for key, item in h5_file[path].items():
        if isinstance(item, h5py.Dataset):
            output[key] = decode_from_hdf5(item[()])
        elif isinstance(item, h5py.Group):
            output[key] = recursively_load_dict_contents_from_group(
                h5_file, path + key + "/"
            )
    return output


def decode_from_hdf5(item: Any) -> Any:
    """
    Decode an item from HDF5 format to python type.

    This currently just converts __none__ to None and some arrays to lists

    :param item: object
        Item to be decoded

    :return  output: object
        Converted input item
    """
    if isinstance(item, str) and item == "__none__":
        output = None
    elif isinstance(item, bytes) and item == b"__none__":
        output = None
    elif isinstance(item, (bytes, bytearray)):
        output = item.decode()
    elif isinstance(item, np.ndarray):
        if item.size == 0:
            output = item
        elif "|S" in str(item.dtype) or isinstance(item[0], bytes):
            output = [it.decode() for it in item]
        else:
            output = item
    elif isinstance(item, np.bool_):
        output = bool(item)
    else:
        output = item
    return output


def recursively_save_dict_contents_to_group(
    h5_file: h5py.File, path: str, dic: Dict[str, Any]
) -> None:
    """
    Recursively save a dictionary to a HDF5 group

    :param h5_file: h5py.File
        Open HDF5 file
    :param path: str
        Path inside the HDF5 file
    :param dic: dict
        The dictionary containing the data
    """
    for key, item in dic.items():
        item = encode_for_hdf5(key, item)
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5_file, path + key + "/", item)
        else:
            h5_file[path + key] = item


def encode_for_hdf5(key: str, item: Any) -> Any:
    """
    Encode an item to a HDF5 saveable format.

    :param item: object
        Object to be encoded, specific options are provided for Bilby types

    :return output: object
        Input item converted into HDF5 saveable format
    """

    if isinstance(item, np.int_):
        item = int(item)
    elif isinstance(item, np.float_):
        item = float(item)
    elif isinstance(item, np.complex_):
        item = complex(item)
    if isinstance(item, (np.ndarray, int, float, complex, str, bytes)):
        output = item
    elif item is None:
        output = "__none__"
    elif isinstance(item, list):
        if len(item) == 0:
            output = item
        elif isinstance(item[0], (str, bytes)) or item[0] is None:
            output = list()
            for value in item:
                if isinstance(value, str):
                    output.append(value.encode("utf-8"))
                elif isinstance(value, bytes):
                    output.append(value)
                else:
                    output.append(b"__none__")
        elif isinstance(item[0], (int, float, complex)):
            output = np.array(item)
        else:
            raise ValueError(f"Cannot save {key}: {type(item)} type")
    elif isinstance(item, pd.DataFrame):
        output = item.to_dict(orient="list")
    elif inspect.isfunction(item) or inspect.isclass(item):
        output = dict(
            __module__=item.__module__, __name__=item.__name__, __class__=True
        )
    elif isinstance(item, dict):
        output = item.copy()
    elif isinstance(item, tuple):
        output = {str(ii): elem for ii, elem in enumerate(item)}
    elif isinstance(item, datetime.timedelta):
        output = item.total_seconds()
    else:
        raise ValueError(f"Cannot save {key}: {type(item)} type")
    return output
