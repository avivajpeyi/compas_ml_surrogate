from typing import List

import numpy as np

from ..compas_output_parser.compas_output import CompasOutput
from .get_total_mass_evolved_per_z import get_total_mass_evolved_per_z


class Universe:
    def __init__(
        self,
        compas_output_path: str,
        m_range: List[float],
        binary_fraction: float,
    ):
        self.path = compas_output_path
        self.compas_output = CompasOutput.from_h5(self.path)
        self.m_range = m_range
        self.binary_fraction = binary_fraction

        (
            self.metallicity_grid,
            self.mass_evolved_per_z,
        ) = self._get_metallicity_grid()

        pass

    @property
    def Mlower(self):
        return self.m_range[0]

    @property
    def Mupper(self):
        return self.m_range[1]

    def _get_metallicity_grid(self):
        # The COMPAS simulation does not evolve all stars
        # give me the correction factor for the total mass evolved
        # I assume each metallicity has the same limits, and does correction
        # factor, but the total mass evolved might be different.
        # This does not change when we change types and other masks this is
        # general to the entire simulation so calculate once

        (
            compas_mass_evolved_per_z,
            compas_z_grid,
        ) = self.compas_output.get_mass_evolved_per_z()
        _, total_mass_evolved_per_z = get_total_mass_evolved_per_z(
            compas_mass_evolved_per_z=compas_mass_evolved_per_z,
            Mlower=self.Mlower,
            Mupper=self.Mupper,
            binaryFraction=self.binary_fraction,
        )
        # Want to recover entire metallicity grid, assume that every metallicity
        # evolved shows in all systems again should not change within same run
        # so dont redo if we reset the data
        return compas_z_grid, total_mass_evolved_per_z
