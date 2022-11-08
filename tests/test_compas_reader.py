import os
import unittest

import numpy as np

from compas_surrogate.compas_output_parser.compas_output import CompasOutput


class TestCompasOutputReader(unittest.TestCase):
    def setUp(self) -> None:
        self.res_dir = os.path.join(
            os.path.dirname(__file__), "test_data", "COMPAS_Output"
        )

    def test_compas_output(self):
        """Test the COMPAS output class can load from h5 and load binary data from BSE_System_Parameters"""
        compas_output = CompasOutput.from_h5(self.res_dir)
        self.assertEqual(compas_output.number_of_systems, 5)
        self.assertEqual(len(compas_output.BSE_System_Parameters), 5)
        binary_0 = compas_output[0]
        self.assertEqual(binary_0, compas_output.get_binary(index=0))
        binary_0_seed = binary_0["SEED"]
        self.assertEqual(
            binary_0, compas_output.get_binary(seed=binary_0_seed)
        )

    def test_html_repr(self):
        compas_output = CompasOutput.from_h5(self.res_dir)
        html = compas_output._repr_html_()
        self.assertIsInstance(html, str)

    def test_get_total_mass_per_z(self):
        compas_output = CompasOutput.from_h5(self.res_dir)
        total_mass_per_z, zs = compas_output.get_mass_evolved_per_z()
        self.assertIsInstance(total_mass_per_z, np.ndarray)
        self.assertEqual(len(zs), len(total_mass_per_z))
