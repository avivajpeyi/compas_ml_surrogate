import os
import unittest

import numpy as np

from compas_surrogate.compas_output_parser.compas_output import CompasOutput
from compas_surrogate.cosmic_integration.universe import Universe


class TestUniverse(unittest.TestCase):
    def setUp(self) -> None:
        self.res_dir = os.path.join(
            os.path.dirname(__file__), "test_data", "COMPAS_Output"
        )
        self.co = CompasOutput.from_h5(self.res_dir)

    def test_universe_total_metalicity(self):
        uni = Universe(self.res_dir, m_range=[15, 150], binary_fraction=0.6)
        uni_total_mass_at_z = uni.mass_evolved_per_z[0]
        compas_total_mass_at_z = self.co.get_mass_evolved_per_z()[0][0]
        self.assertGreater(uni_total_mass_at_z, compas_total_mass_at_z)
