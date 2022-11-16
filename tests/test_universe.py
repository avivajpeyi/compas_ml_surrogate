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
        uni = Universe(self.res_dir)
        uni.run()
        print("run complete")