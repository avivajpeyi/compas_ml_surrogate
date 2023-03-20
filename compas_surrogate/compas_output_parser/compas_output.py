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
    def __init__(self, h5:h5py.File, h5_filename:str=""):
        self.h5 = h5
        self.h5_filename = h5_filename


    def printSummary(self, h5name=None, h5file=None, excludeList=""):
        """LIFTED FROM COMPAS UTILS"""
        h5name = self.h5
        h5file = h5py.File(h5name, "r")
        ok = True  # result

        try:

            mtime = os.path.getmtime(h5name)  # time file last modified
            lastModified = datetime.datetime.fromtimestamp(
                mtime
            )  # ... formatted

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
                        if (
                            h5file[group][column].shape[0]
                            > groupMaxEntries[gIdx]
                        ):
                            groupMaxEntries[gIdx] = h5file[group][
                                column
                            ].shape[0]

            for widthColumns in range(
                10
            ):  # better not be more than 10**10 columns!
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
                    ).format(
                        "COMPAS Filename", "Columns", "Entries", "Unique SEEDs"
                    )
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
                if (
                    not Run_Details_Filename in excludeList
                ):  # ... if not excluded
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
                                    list(h5file[Run_Details_Filename].keys())[
                                        0
                                    ]
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
                        uniqueSeedsStr = str(
                            len(np.unique(h5file[group]["SEED"]))
                        )
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
            print(
                "printSummary: Error accessing HDF5 file", h5name, ":", str(e)
            )
            ok = False

    def __repr__(self):
        with Capturing() as o:
            self.printSummary()
        return "\n".join(o)

    @classmethod
    def from_hdf5(cls, h5name: str, ):
        return cls(h5py(h5name, 'r'), h5name)




#
# class CompasOutput:
#     def __init__(
#             self,
#             outdir: str,
#             Run_Details: Dict[str, List],
#             BSE_System_Parameters: pd.DataFrame,
#             BSE_Supernovae: Optional[pd.DataFrame] = None,
#             BSE_Common_Envelopes: Optional[pd.DataFrame] = None,
#             BSE_RLOF: Optional[pd.DataFrame] = None,
#             BSE_Double_Compact_Objects: Optional[pd.DataFrame] = None,
#     ):
#         """
#
#         :param outdir: Directory where the COMPAS output is stored (should contain COMPAS_Output.h5)
#         :type outdir: str
#         :param Run_Details: Configs for the run
#         :type Run_Details: Dict[str, List]
#         :param BSE_System_Parameters: All system parameters with one unique row per binary (i.e. one row per 'SEED')
#         :type BSE_System_Parameters: pd.DataFrame
#         :param BSE_Supernovae: Supernovae data for the run
#         :type BSE_Supernovae: Optional[pd.DataFrame]
#         :param BSE_Common_Envelopes: Common Envelope data for the run
#         :type BSE_Common_Envelopes: Optional[pd.DataFrame]
#         :param BSE_RLOF: Roche Lobe Overflow data for the run
#         :type BSE_RLOF: Optional[pd.DataFrame]
#         """
#
#         self.outdir = outdir
#         self.Run_Details = Run_Details
#         self.BSE_Supernovae = BSE_Supernovae
#         self.BSE_System_Parameters = BSE_System_Parameters
#         self.BSE_Common_Envelopes = BSE_Common_Envelopes
#         self.BSE_RLOF = BSE_RLOF
#         self.BSE_Double_Compact_Objects = BSE_Double_Compact_Objects
#         self.number_of_systems = self.Run_Details["number-of-systems"]
#         self.detailed_output_exists = (
#             True if self.Run_Details["detailed-output"] == 1 else False
#         )
#
#     def __getitem__(self, item):
#         return self.get_binary(index=item)
#
#     def get_binary(self, index=None, seed=None) -> Dict:
#         """
#         Get a binary by row index or seed
#         :param index: int row index
#         :param seed: int unique binary seed
#         """
#         all_seeds = self.BSE_System_Parameters.SEED
#         if index is not None:
#             seed = all_seeds.iloc[index]
#         else:
#             index = self.BSE_System_Parameters[all_seeds == seed].index[0]
#         data = dict(index=index, SEED=seed)
#         if self.detailed_output_exists:
#             det_fn = os.path.join(
#                 self.outdir,
#                 "Detailed_Output",
#                 f"BSE_Detailed_Output_{index}.h5",
#             )
#             data["detailed_output"] = pd.DataFrame(parse_h5_file(det_fn))
#         for key in self.__dict__:
#             val = self.__dict__[key]
#             if isinstance(val, pd.DataFrame):
#                 val = val[val["SEED"] == seed]
#                 if len(val) == 0:
#                     data[key] = None
#                 elif len(val) > 1:
#                     raise ValueError("Data for two binaries with same key")
#                 else:
#                     data[key] = val.to_dict("records")[0]
#
#         return data
#
#     @staticmethod
#     def stellar_type(df, key) -> np.ndarray:
#         """Stellar types in dataframe
#
#         :return np.ndarray[[StellarType, StellarType]]: each row is a binary, each column is a star
#         """
#         types = np.array([df[f"{key}(1)"].values, df[f"{key}(2)"].values]).T
#         # convert to StellarType
#         return np.vectorize(StellarType)(types)
#
#     @classmethod
#     def from_h5(cls, h5path):
#         """
#         Loads a COMPAS output file from h5 format.
#
#         :param h5path: the directory where the output files are written
#         :return: a COMPASOutput object
#         """
#         # check that the file extension is .h5:
#         if not h5path.endswith(".h5"):
#             raise ValueError("COMPAS output file must be in h5 format")
#
#         data = parse_h5_file(h5path)
#         run_details = data["Run_Details"]
#         keys = data.keys()
#         logging.debug(f"COMPAS run with keys: {keys}")
#         new_data = {}
#         for k in keys:
#             logging.debug(f"Reading {k}")
#             try:
#                 new_data[k] = pd.DataFrame(data[k])
#             except Exception as e:
#                 logging.debug(f"Failed to read {k} with error {e}")
#         new_data["Run_Details"] = pd.DataFrame(run_details).to_dict("records")[
#             0
#         ]
#         new_data["outdir"] = h5path
#         return cls(**new_data)
#
#     def __repr__(self):
#         rep = ["COMPAS OUTPUT"]
#         for k, v in self.__dict__.items():
#             att_rep = f"--{k}: {v},"
#             if isinstance(v, str):
#                 pass
#             elif hasattr(v, "__len__"):
#                 att_rep = f"--{k}: {len(v)},"
#             if hasattr(v, "shape"):
#                 att_rep = f"--{k}: shape {v.shape},"
#             rep.append(att_rep)
#         return "\n".join(rep)
#
#     def _repr_html_(self):
#         dfs = {
#             k: v
#             for k, v in self.__dict__.items()
#             if isinstance(v, pd.DataFrame)
#         }
#         dfs["Run_Details"] = pd.DataFrame(self.Run_Details, index=[0]).T
#         dfs["Run_Details"].columns = ["Value"]
#         dfs["Run_Details"].index.name = "Setting"
#
#         elemnts = []
#         for k, v in self.__dict__.items():
#             if k in dfs.keys():
#                 v = dfs[k]._repr_html_()
#
#             elemnts.append(
#                 element_template.format(
#                     group_id=k + str(uuid.uuid4()),
#                     group=k,
#                     xr_data=v,
#                 )
#             )
#
#         formatted_html_template = html_template.format("".join(elemnts))
#         css_template = css  # pylint: disable=possibly-unused-variable
#         html_repr = (
#             f"{locals()['formatted_html_template']}{locals()['css_template']}"
#         )
#
#         return html_repr
#
#     @property
#     def initial_z(self) -> np.ndarray:
#         """
#         Returns the initial metallicity of the binary population
#         """
#         return self.BSE_System_Parameters["Metallicity@ZAMS(1)"].unique()
#
#     def get_mass_evolved_per_z(self) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Returns the mass evolved per metallicity and the metallicity bins
#         :return: mass evolved per metallicity, metallicity bins
#         """
#         all_sys = self.BSE_System_Parameters
#         all_metals = all_sys["Metallicity@ZAMS(1)"]
#         m1s, m2s = all_sys["Mass@ZAMS(1)"], all_sys["Mass@ZAMS(2)"]
#         total = []
#         for Z in self.initial_z:
#             mask = all_metals == Z
#             total.append(np.sum(m1s[mask]) + np.sum(m2s[mask]))
#         return np.array(total), self.initial_z
#
#     def get_mask(
#             self,
#             type="BBH",
#             withinHubbleTime=True,
#             pessimistic=True,
#             noRLOFafterCEE=True,
#     ) -> np.ndarray:
#         type = DCOType[type]
#         dco_stellar_types = self.stellar_type(
#             self.BSE_Double_Compact_Objects, "Stellar_Type"
#         )
#         hubble_flag = self.BSE_Double_Compact_Objects["Merges_Hubble_Time"]
#         dco_seeds = self.BSE_Double_Compact_Objects["SEED"]
#
#         return self.BSE_System_Parameters["SEED"] == dco_seeds
