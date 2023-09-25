from typing import Callable, Union

import numpy as np


def ln_poisson_likelihood(n_obs: int, n_model: int, ignore_factorial=True) -> float:
    """
    Computes LnL(N_obs | N_model) = N_obs * ln(N_model) - N_model - ln(N_obs!)

    :param n_obs: number of observed events
    :param n_model: number of events predicted by the model
    :param ignore_factorial: ignore the factorial term in the likelihood
    :return: the log likelihood
    """
    if n_model <= 0:
        return -np.inf
    lnl = n_obs * np.log(n_model) - n_model

    if ignore_factorial is False:
        lnl += -np.log(np.math.factorial(n_obs))

    return lnl


def ln_mcz_grid_likelihood(mcz_obs: np.ndarray, model_prob_func: Callable) -> float:
    """
    Computes LnL(mc, z | model) = sum_i  ln p(mc_i, z_i | model)     (for N_obs events)
    :param mcz_obs: [[mc,z], [mc,z], ...] Array of observed mc and z values for each event (exact measurement)
    :param model_prob_func: model_func(mc,z) -> prob(mc_i, z_i | model)
    :return:
    """
    return np.sum([np.log(model_prob_func(mc, z)) for mc, z in mcz_obs])


def ln_likelihood(
    mcz_obs: np.ndarray,
    model_prob_func: Callable,
    n_model: float,
    detailed=False,
) -> Union[float, tuple]:
    poisson_lnl = ln_poisson_likelihood(len(mcz_obs), n_model)
    mcz_lnl = ln_mcz_grid_likelihood(mcz_obs, model_prob_func)
    lnl = poisson_lnl + mcz_lnl
    if detailed:
        return lnl, poisson_lnl, mcz_lnl
    else:
        return lnl
