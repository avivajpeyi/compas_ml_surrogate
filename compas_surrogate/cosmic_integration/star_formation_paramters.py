from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.stats import qmc
from scipy.stats.qmc import LatinHypercube

DEFAULT_SF_PARAMETERS = dict(
    aSF=0.01,
    bSF=2.77,
    cSF=2.9,
    dSF=4.7,
    muz=-0.23,
    sigma0=0.39,
)

STAR_FORMATION_RANGES = dict(
    muz=[-0.5, -0.001],  # Jeff's alpha
    sigma0=[0.1, 0.6],  # Jeff's sigma
    aSF=[0.005, 0.015],
    dSF=[4.2, 5.2],
)


def draw_star_formation_samples(
    n=1000, parameters=None, as_list=False
) -> Union[Dict[str, np.ndarray], List[Dict]]:
    """Draw samples from the star formation parameters."""
    if parameters is None:
        parameters = ["muz", "sigma0", "aSF", "dSF"]
    assert all(
        [p in STAR_FORMATION_RANGES for p in parameters]
    ), "Invalid parameters"
    num_dim = len(parameters)
    sampler = LatinHypercube(d=num_dim)
    samples = sampler.random(n)
    parameter_ranges = np.array([STAR_FORMATION_RANGES[p] for p in parameters])
    lower_bound = parameter_ranges[:, 0]
    upper_bound = parameter_ranges[:, 1]
    scaled_samples = qmc.scale(
        samples, l_bounds=lower_bound, u_bounds=upper_bound
    )
    dict_of_params = {
        p: scaled_samples[:, i] for i, p in enumerate(parameters)
    }
    if as_list:
        return [
            dict(zip(dict_of_params, t)) for t in zip(*dict_of_params.values())
        ]
