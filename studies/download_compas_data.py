from compas_surrogate.utils import download_file

ZENODO_LINK = "https://zenodo.org/record/6346444/files/Z_all.zip?download=1"


def main():
    download_file(ZENODO_LINK, "Z_all.zip")


if __name__ == "__main__":
    main()
