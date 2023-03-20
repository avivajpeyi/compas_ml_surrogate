import os

import pytest

HERE = os.path.dirname(__file__)
TEST_DIR = os.path.join(HERE, "test_data")
SMALL_FILE = os.path.join(TEST_DIR, "COMPAS_Output/COMPAS_Output.h5")
LARGE_FILE = os.path.join(TEST_DIR, "Z_all.h5")


@pytest.fixture
def test_datapath():
    """Test data."""
    if os.path.exists(LARGE_FILE):
        return LARGE_FILE
    return SMALL_FILE
