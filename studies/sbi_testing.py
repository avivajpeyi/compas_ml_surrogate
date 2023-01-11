import numpy as np
from bilby.core.prior import PriorDict, Uniform
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

TRUE_MODEL = lambda x, m, c: m * x + c
TRUE_PARAMS = [1.0, -3.0]  # m, c

NOISE_SIGMA = 0.1


def generate_data():
    pass


def prior():
    pass


def simulation(m, c):
    pass


def simulate_posterior():
    pass


def plot():
    pass
