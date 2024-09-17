from typing import Callable

import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.optimize import minimize


def Dchi2(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    model: Callable,
    args: list = None,
    verbose: bool = False,
):
    """
    Compute the chi2 improvement with certain model[Callable]

    the model should be a function that takes (x, *args) and return (y_model)

    """
    chi2_orig = np.sum(y**2 / yerr**2)

    def chi2(_args):
        y_model = model(x, *_args)
        return np.sum((y - y_model) ** 2 / yerr**2)

    res = minimize(chi2, args, method="Nelder-Mead")
    if verbose:
        print("Chi-square is reduced by", res.fun, "with extra d.o.f. =", len(args))
    return chi2_orig - res.fun
