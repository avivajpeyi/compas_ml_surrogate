from compas_surrogate.compas_surrogate_version import __version__


def test_version():
    assert __version__ is not None


if __name__ == "__main__":
    test_version()
