import datetime
import logging
import os
import sys
import uuid
from io import StringIO
from typing import Dict, List, Optional, Tuple

import h5py
import h5py as h5
import numpy as np
import pandas as pd

from .h5_parser import parse_h5_file
from .html_templates import css, element_template, html_template
from .types import DCOType, StellarType


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


logging.getLogger().setLevel(logging.DEBUG)

Run_Details_Filename = "Run_Details"


class CompasOutput:
    def __init__(self, h5: str = ""):
        self.h5 = h5

    def printSummary(self, excludeList=""):
        """LIFTED FROM COMPAS UTILS"""
        h5name = self.h5
        h5file = h5py.File(h5name, "r")
        ok = True  # result

        try:

            mtime = os.path.getmtime(h5name)  # time file last modified
            lastModified = datetime.datetime.fromtimestamp(mtime)  # ... formatted

            fileSize = os.path.getsize(h5name)  # file size (in bytes)
            strFileSize = ("{:<11.4f}").format(
                fileSize / 1024.0 / 1024.0 / 1024.0
            )  # ... formatted in GB

            print("\n\nSummary of HDF5 file", h5name)
            print("=" * (21 + len(h5name)))

            print("\nFile size    :", strFileSize.strip(), "GB")
            print("Last modified:", lastModified)

            # get widths for columns to be displayed - it's a bit more work,
            # and a bit redundant, but it is neater... (and we don't have thousands of files...)
            maxFilenameLen = -1
            maxColumns = -1
            groupMaxEntries = []

            keyList = list(h5file.keys())
            detailedOutput = isinstance(
                h5file[keyList[0]], h5.Dataset
            )  # processing detailed output file?

            if detailedOutput:  # detailed output file

                maxColumns = len(keyList)
                groupMaxEntries.append(-1)
                for dIdx, dataset in enumerate(h5file.keys()):
                    if h5file[dataset].shape[0] > groupMaxEntries[0]:
                        groupMaxEntries[0] = h5file[dataset].shape[0]

            else:  # assume COMPAS_Output file

                for gIdx, group in enumerate(h5file.keys()):

                    groupMaxEntries.append(-1)

                    if len(group) > maxFilenameLen:
                        maxFilenameLen = len(group)
                    if len(h5file[group].keys()) > maxColumns:
                        maxColumns = len(h5file[group].keys())

                    columns = h5file[group].keys()
                    for idx, column in enumerate(columns):
                        if h5file[group][column].shape[0] > groupMaxEntries[gIdx]:
                            groupMaxEntries[gIdx] = h5file[group][column].shape[0]

            for widthColumns in range(10):  # better not be more than 10**10 columns!
                if maxColumns < 10**widthColumns:
                    break  # 'columns' width

            maxEntries = max(groupMaxEntries)
            for widthEntries in range(
                10
            ):  # better not be more than 10**10 entries per column!
                if maxEntries < 10**widthEntries:
                    break  # 'entries' width

            if detailedOutput:  # detailed output file

                print(
                    (
                        "\n{:<"
                        + str(max(7, widthColumns))
                        + "}   {:<"
                        + str(max(7, widthEntries))
                        + "}"
                    ).format("Columns", "Entries")
                )
                print(
                    "-" * (max(7, widthColumns)),
                    " ",
                    "-" * (max(7, widthEntries)),
                )

                print(
                    (
                        "{:>"
                        + str(max(7, widthColumns))
                        + "}   {:>"
                        + str(max(7, widthEntries))
                        + "}"
                    ).format(len(keyList), groupMaxEntries[0])
                )

            else:  # assume COMPAS_Output file

                print(
                    (
                        "\n{:<"
                        + str(maxFilenameLen)
                        + "}   {:<"
                        + str(max(7, widthColumns))
                        + "}   {:<"
                        + str(max(7, widthEntries))
                        + "}   {:<"
                        + str(max(12, widthEntries))
                        + "}"
                    ).format("COMPAS Filename", "Columns", "Entries", "Unique SEEDs")
                )
                print(
                    "-" * (maxFilenameLen),
                    " ",
                    "-" * (max(7, widthColumns)),
                    " ",
                    "-" * (max(7, widthEntries)),
                    " ",
                    "-" * (max(12, widthEntries)),
                )

                # do Run_Details file first
                if not Run_Details_Filename in excludeList:  # ... if not excluded
                    print(
                        (
                            "{:<"
                            + str(maxFilenameLen)
                            + "}   {:>"
                            + str(max(7, widthColumns))
                            + "}   {:>"
                            + str(max(7, widthEntries))
                            + "}"
                        ).format(
                            Run_Details_Filename,
                            len(h5file[Run_Details_Filename].keys()),
                            len(
                                h5file[Run_Details_Filename][
                                    list(h5file[Run_Details_Filename].keys())[0]
                                ]
                            ),
                        )
                    )

                # do remaining files (groups)
                for gIdx, group in enumerate(h5file.keys()):
                    if group in excludeList:
                        continue  # skip if excluded
                    if group == Run_Details_Filename:
                        continue  # Run_details already done (or excluded)

                    try:
                        uniqueSeedsStr = str(len(np.unique(h5file[group]["SEED"])))
                    except Exception as e:
                        uniqueSeedsStr = " "

                    print(
                        (
                            "{:<"
                            + str(maxFilenameLen)
                            + "}   {:>"
                            + str(max(7, widthColumns))
                            + "}   {:>"
                            + str(max(7, widthEntries))
                            + "}   {:>"
                            + str(max(12, widthEntries))
                            + "}"
                        ).format(
                            group,
                            len(h5file[group].keys()),
                            groupMaxEntries[gIdx],
                            uniqueSeedsStr,
                        )
                    )

            print("\n")

        except Exception as e:  # error occurred accessing the input file
            print("printSummary: Error accessing HDF5 file", h5name, ":", str(e))
            ok = False

    def __repr__(self):
        with Capturing() as o:
            self.printSummary()
        return "\n".join(o)

    def _repr_html_(self):
        return self.__repr__()
