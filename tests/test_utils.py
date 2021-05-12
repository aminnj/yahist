import numpy as np

from yahist import Hist1D, Hist2D, utils


import os

np.set_printoptions(linewidth=120)

import pytest


def allclose(a, b, equal_nan=False):
    return np.testing.assert_allclose(np.array(a), np.array(b), equal_nan=equal_nan)


def test_is_listlike():
    assert utils.is_listlike([1, 2, 3])
    assert utils.is_listlike((1, 2, 3))
    assert utils.is_listlike(np.arange(3))
    assert not utils.is_listlike(3.0)


def test_has_uniform_spacing():
    assert utils.has_uniform_spacing([1, 2, 3, 4])
    assert utils.has_uniform_spacing(np.arange(10))
    assert utils.has_uniform_spacing(np.arange(10) - 5)
    assert not utils.has_uniform_spacing([1, 2, 3, 5])
    assert not utils.has_uniform_spacing([-4, -1, 1, 4, 5])
    assert not utils.has_uniform_spacing([-6, 0, 1, 5, 6])

    a = np.linspace(0, 10, 100)
    assert utils.has_uniform_spacing(a)
    a[-1] += 0.01
    assert not utils.has_uniform_spacing(a)


def test_clopper_pearson_error():
    passed = np.array([0, 2, 4, 2])
    total = np.array([4, 4, 4, 4])
    level = np.array([0.68, 0.68, 0.68, 0.95])
    lows, highs = utils.clopper_pearson_error(passed, total, level=level)
    allclose(lows, [np.nan, 0.18621103, 0.63245553, 0.06758599], equal_nan=True)
    allclose(highs, [0.36754447, 0.81378897, np.nan, 0.93241401], equal_nan=True)


def test_poisson_errors():
    lows, highs = utils.poisson_errors(np.array([0, 1, 5, 100]))
    allclose(lows, [0.0, 0.17274753, 2.84027568, 90.01654275])
    allclose(highs, [1.84105476, 3.29956971, 8.38253922, 111.03359339])


def test_binomial_obs_z():
    data = np.array([5, 10])
    bkg = np.array([2.0, 15.0])
    bkgerr = np.array([1.0, 1.0])
    z = utils.binomial_obs_z(data, bkg, bkgerr)
    allclose(z, [1.05879838, -1.44619545])


def test_nan_to_num():
    @utils.ignore_division_errors
    def f(x):
        return (np.array(x) * 0) / 0.0

    g = utils.nan_to_num(f)
    allclose(f([1]), [np.nan], equal_nan=True)
    allclose(g([1]), [0.0], equal_nan=True)


if __name__ == "__main__":
    pytest.main(["--capture=no", __file__])
