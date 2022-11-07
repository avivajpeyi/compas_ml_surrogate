import os
import pandas as pd
from .h5_parser import parse_h5_file
from typing import Optional, Dict, List
import uuid

from .html_templates import html_template, element_template, css


class CompasOutput:
    def __init__(
            self,
            outdir: str,
            Run_Details: Dict[str, List],
            BSE_System_Parameters: pd.DataFrame,
            BSE_Supernovae: Optional[pd.DataFrame] = None,
            BSE_Common_Envelopes: Optional[pd.DataFrame] = None,
            BSE_RLOF: Optional[pd.DataFrame] = None,
    ):
        """

        :param outdir: Directory where the COMPAS output is stored (should contain COMPAS_Output.h5)
        :type outdir: str
        :param Run_Details: Configs for the run
        :type Run_Details: Dict[str, List]
        :param BSE_System_Parameters: All system parameters with one unique row per binary (i.e. one row per 'SEED')
        :type BSE_System_Parameters: pd.DataFrame
        :param BSE_Supernovae: Supernovae data for the run
        :type BSE_Supernovae: Optional[pd.DataFrame]
        :param BSE_Common_Envelopes: Common Envelope data for the run
        :type BSE_Common_Envelopes: Optional[pd.DataFrame]
        :param BSE_RLOF: Roche Lobe Overflow data for the run
        :type BSE_RLOF: Optional[pd.DataFrame]
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

    def get_binary(self, index=None, seed=None) -> Dict:
        """
        Get a binary by row index or seed
        :param index: int row index
        :param seed: int unique binary seed
        """
        all_seeds = self.BSE_System_Parameters.SEED
        if index is not None:
            seed = all_seeds.iloc[index]
        else:
            index = self.BSE_System_Parameters[all_seeds == seed].index[0]
        data = dict(index=index, SEED=seed)
        if self.detailed_output_exists:
            det_fn = os.path.join(self.outdir, "Detailed_Output", f"BSE_Detailed_Output_{index}.h5")
            data["detailed_output"] = pd.DataFrame(parse_h5_file(det_fn))
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
    def from_h5(cls, outdir):
        """
        Loads a COMPAS output file from h5 format.

        :param outdir: the directory where the output files are written
        :return: a COMPASOutput object
        """
        filename = os.path.join(outdir, 'COMPAS_Output.h5')
        data = parse_h5_file(filename)
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

    def _repr_html_(self):
        dfs = {k:v for k, v in self.__dict__.items() if isinstance(v, pd.DataFrame)}
        dfs['Run_Details'] = pd.DataFrame(self.Run_Details, index=[0]).T
        dfs['Run_Details'].columns = ['Value']
        dfs['Run_Details'].index.name = 'Setting'

        elemnts = []
        for k, v in self.__dict__.items():
            if k in dfs.keys():
                v  = dfs[k]._repr_html_()

            elemnts.append(element_template.format(
                group_id=k + str(uuid.uuid4()),
                group=k,
                xr_data=v,
            ))

        formatted_html_template = html_template.format("".join(elemnts))
        css_template = css  # pylint: disable=possibly-unused-variable
        html_repr = f"{locals()['formatted_html_template']}{locals()['css_template']}"

        return html_repr
