import codecs
import os
import re

from setuptools import find_packages, setup

NAME = "compas_surrogate"
PACKAGES = find_packages(where=".")
META_PATH = os.path.join("compas_surrogate", "__init__.py")
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
]
INSTALL_REQUIRES = [
    "scikit-image",
    "astropy",
    "matplotlib",
    "h5py",
    "plotly",
    "loguru",
    "tqdm",
    "requests",
    "george",
    "scikit-learn",
    "arviz",
    "fpdf",
]
EXTRA_REQUIRE = {
    "gpflow": ["gpflow"],
    "test": ["pytest>=3.6"],
}
EXTRA_REQUIRE["dev"] = EXTRA_REQUIRE["test"] + [
    "pre-commit",
    "flake8",
    "black<=21.9b0",
    "isort",
    "ipython-autotime",
    "memory_profiler",
]

HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        use_scm_version={
            "write_to": os.path.join(NAME, f"{NAME}_version.py"),
            "write_to_template": '__version__ = "{version}"\n',
        },
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        url=find_meta("uri"),
        license=find_meta("license"),
        description=find_meta("description"),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_data={},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRE,
        classifiers=CLASSIFIERS,
        zip_safe=True,
        entry_points={
            "console_scripts": [
                "make_detection_matrices = compas_surrogate.data_generation.cli:cli_matrix_generation",
                "compile_matrix_h5 = compas_surrogate.data_generation.cli:cli_compile_h5",
            ]
        },
    )
