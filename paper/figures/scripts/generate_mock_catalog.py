import os

import requests
from tqdm.auto import tqdm

from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.logger import logger

CLEAN = True
DATA_LINK = (
    "https://sandbox.zenodo.org/record/1145903/files/uni.npz?download=1"
)
DATA_FILE = "uni.npz"


def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        pbar = tqdm(
            unit="B",
            total=int(r.headers["Content-Length"]),
            desc="Downloading data",
        )
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)
    return filename


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
