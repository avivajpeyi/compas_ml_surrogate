from multiprocessing import cpu_count

import requests
from tqdm.auto import tqdm


def download_file(url: str, filename: str) -> str:
    """
    Download a file from a given url and save it to a given filename.
    :param url: download URL
    :param filename: Download fname to save to
    :return: None
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


def get_num_workers():
    num_workers = cpu_count()
    if num_workers > 64:
        num_workers = 16
    elif num_workers < 16:
        num_workers = 4
    return num_workers
