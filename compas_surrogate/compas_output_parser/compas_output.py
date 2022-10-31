import os
import h5py
import pandas as pd
from .h5utils import recursively_load_dict_contents_from_group
from typing import Optional


class CompasOutput:
    def __init__(
            self,
            outdir: str,
            Run_Details: Optional[dict] = None,
            BSE_Supernovae: Optional[pd.DataFrame] = None,
            BSE_System_Parameters: Optional[pd.DataFrame] = None,
            BSE_Common_Envelopes: Optional[pd.DataFrame] = None,
            BSE_RLOF: Optional[pd.DataFrame] = None,
    ):
        """

        :param outdir:
        :param Run_Details: Configs for the run
        :param BSE_Supernovae: pd.DataFrame of supernovae
        :param BSE_System_Parameters: pd.DataFrame of system parameters
        """
        self.outdir = outdir
        self.Run_Details = Run_Details
        self.BSE_Supernovae = BSE_Supernovae
        self.BSE_System_Parameters = BSE_System_Parameters
        self.BSE_Common_Envelopes = BSE_Common_Envelopes
        self.BSE_RLOF = BSE_RLOF
        self.number_of_systems = self.Run_Details['number-of-systems']
        self.detailed_output_exists = True if self.Run_Details["detailed-output"] == 1 else False

    def __getitem__(self, item):
        return self.get_binary(index=item)

    def get_binary(self, index=None, seed=None):
        all_seeds = self.BSE_System_Parameters.SEED
        if index is not None:
            seed = all_seeds.iloc[index]
        else:
            index = self.BSE_System_Parameters[all_seeds == seed].index[0]
        data = {}
        if self.detailed_output_exists:
            det_fn = os.path.join(self.outdir, "Detailed_Output", f"BSE_Detailed_Output_{index}.h5")
            data["detailed_output"] = load_compas_detailed_output(det_fn)
        for key in self.__dict__:
            val = self.__dict__[key]
            if isinstance(val, pd.DataFrame):
                val = val[val['SEED'] == seed]
                if len(val) == 0:
                    data[key] = None
                elif len(val) > 1:
                    raise ValueError("Data for two binaries with same key")
                else:
                    data[key] = val.to_dict('records')[0]

        return data

    @classmethod
    def from_hdf5(cls, outdir):
        """
        Loads a COMPAS output file from HDF5 format.

        :param outdir: the directory where the output files are written
        :return: a COMPASOutput object
        """

        filename = os.path.join(outdir, 'COMPAS_Output.h5')
        with h5py.File(filename, "r") as ff:
            data = recursively_load_dict_contents_from_group(ff, '/')
        run_details = data['Run_Details']
        for k in data.keys():
            data[k] = pd.DataFrame(data[k])
        data['Run_Details'] = pd.DataFrame(run_details).to_dict('records')[0]
        data['outdir'] = outdir
        return cls(**data)

    def __repr__(self):
        rep = ["COMPAS OUTPUT"]
        for k, v in self.__dict__.items():
            att_rep = f"--{k}: {v},"
            if isinstance(v, str):
                pass
            elif hasattr(v, "__len__"):
                att_rep = f"--{k}: {len(v)},"
            if hasattr(v, "shape"):
                att_rep = f"--{k}: shape {v.shape},"
            rep.append(att_rep)
        return "\n".join(rep)


def load_compas_detailed_output(filename):
    with h5py.File(filename, "r") as ff:
        data = recursively_load_dict_contents_from_group(ff, '/')
    return pd.DataFrame(data)
