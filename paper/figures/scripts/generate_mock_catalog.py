import os

from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.logger import logger
from compas_surrogate.utils import download_file

CLEAN = True
DATA_LINK = (
    "https://sandbox.zenodo.org/record/1145903/files/uni.npz?download=1"
)
DATA_FILE = "uni.npz"


def main():
    if not os.path.exists(DATA_FILE) or CLEAN:
        logger.info("Downloading")
        download_file(DATA_LINK, DATA_FILE)
    uni = Universe.from_npz(DATA_FILE)
    mock_cat = uni.sample_possible_event_matrix()
    fname = "../mock_events.png"
    mock_cat.plot(save=False).savefig(fname)
    logger.success(f"Plot saved to {fname}")


if __name__ == "__main__":
    main()
