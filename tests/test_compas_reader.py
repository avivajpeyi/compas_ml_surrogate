import os
import unittest

import numpy as np

from compas_surrogate.compas_output_parser.compas_output import CompasOutput


class TestCompasOutputReader(unittest.TestCase):
    def setUp(self) -> None:
        self.res_dir = os.path.join(
            os.path.dirname(__file__),
            "test_data",
            "COMPAS_Output/COMPAS_Output.h5",
        )
        self.res_dir = "/Users/avaj0001/Documents/projects/compas_dev/quasir_compass_blocks/data/COMPAS_Output.h5"
        self.co = CompasOutput.from_h5(self.res_dir)
        self.num_sys = 500

    def test_compas_output_basics(self):
        self.assertEqual(self.co.number_of_systems, self.num_sys)
        self.assertEqual(len(self.co.BSE_System_Parameters), self.num_sys)
        binary_0 = self.co[0]
        self.assertEqual(binary_0, self.co.get_binary(index=0))
        binary_0_seed = binary_0["SEED"]
        self.assertEqual(binary_0, self.co.get_binary(seed=binary_0_seed))

    def test_html_repr(self):
        print("Loaded file, now making html")
        html = self.co._repr_html_()
        self.assertIsInstance(html, str)

    def test_get_total_mass_per_z(self):
        total_mass_per_z, zs = self.co.get_mass_evolved_per_z()
        self.assertIsInstance(total_mass_per_z, np.ndarray)
        self.assertEqual(len(zs), len(total_mass_per_z))

    def test_make_mask(self):
        mask = self.co.make_mask()
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (self.num_sys,))
