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


def pair(t1: np.array, t2: np.array, tol: float = 0.01) -> tuple[np.array, np.array, np.array]:
    """
    INPUT :
        t1, t2 [np.array] - two sorted array to be paired
        tol[float] - tolerance for pairing


    OUTPUT: the indices of the pairs w.r.t. t1 and t2.
    """
    ind1 = []
    ind2 = []
    diff = []

    i1 = 0
    i2 = 0

    while i1 < len(t1) and i2 < len(t2):
        if t2[i2] - t1[i1] > tol:
            i1 += 1
        elif t1[i1] - t2[i2] > tol:
            i2 += 1
        else:
            ind1.append(i1)
            ind2.append(i2)
            diff.append(t1[i1] - t2[i2])
            i1 += 1
    return np.array(ind1), np.array(ind2), np.array(diff)
