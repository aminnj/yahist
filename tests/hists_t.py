import unittest

import numpy as np
from yahist import Hist1D, Hist2D, utils
import os

np.set_printoptions(linewidth=120)


class UtilsTest(unittest.TestCase):
    def test_is_listlike(self):
        self.assertTrue(utils.is_listlike([1, 2, 3]))
        self.assertTrue(utils.is_listlike((1, 2, 3)))
        self.assertTrue(utils.is_listlike(np.arange(3)))
        self.assertFalse(utils.is_listlike(3.0))

    def test_has_uniform_spacing(self):
        self.assertTrue(utils.has_uniform_spacing([1, 2, 3, 4]))
        self.assertFalse(utils.has_uniform_spacing([1, 2, 3, 5]))

    def test_clopper_pearson_error(self):
        passed = np.array([0, 2, 4, 2])
        total = np.array([4, 4, 4, 4])
        level = np.array([0.68, 0.68, 0.68, 0.95])
        lows, highs = utils.clopper_pearson_error(passed, total, level=level)
        self.assertTrue(
            np.allclose(
                lows,
                np.array([np.nan, 0.18621103, 0.63245553, 0.06758599]),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                highs,
                np.array([0.36754447, 0.81378897, np.nan, 0.93241401]),
                equal_nan=True,
            )
        )

    def test_poisson_errors(self):
        lows, highs = utils.poisson_errors(np.array([0, 1, 5, 100]))
        self.assertTrue(
            np.allclose(lows, np.array([0.0, 0.17274753, 2.84027568, 90.01654275]))
        )
        self.assertTrue(
            np.allclose(
                highs, np.array([1.84105476, 3.29956971, 8.38253922, 111.03359339])
            )
        )

    def test_binomial_obs_z(self):
        data = np.array([5, 10])
        bkg = np.array([2.0, 15.0])
        bkgerr = np.array([1.0, 1.0])
        z = utils.binomial_obs_z(data, bkg, bkgerr)
        self.assertTrue(np.allclose(z, np.array([1.05879838, -1.44619545])))

    def test_nan_to_num(self):
        @utils.ignore_division_errors
        def f(x):
            return (x * 0) / 0.0

        g = utils.nan_to_num(f)
        self.assertTrue(
            np.allclose(f(np.array([1])), np.array([np.nan]), equal_nan=True)
        )
        self.assertTrue(np.allclose(g(np.array([1])), np.array([0.0]), equal_nan=True))

    def test_expr_to_lambda(self):
        f = utils.expr_to_lambda("x+a+a+b+np.pi")
        g = lambda x, a, b: x + a + a + b + np.pi
        self.assertEqual(f(1, 2, 3), g(1, 2, 3))

        f = utils.expr_to_lambda("m*x+b")
        g = lambda x, m, b: m * x + b
        self.assertEqual(f(1, 2, 3), g(1, 2, 3))

        f = utils.expr_to_lambda("1+np.poly1d([a,b,c])(x)")
        g = lambda x, a, b, c: 1 + np.poly1d([a, b, c])(x)
        self.assertEqual(f(1, 2, 3, 4), g(1, 2, 3, 4))

        f = utils.expr_to_lambda("const + norm*np.exp(-(x-mu)**2/(2*sigma**2))")
        g = lambda x, const, norm, mu, sigma: const + norm * np.exp(
            -((x - mu) ** 2) / (2 * sigma ** 2)
        )
        self.assertEqual(f(1, 2, 3, 4, 5), g(1, 2, 3, 4, 5))


class FitTest(unittest.TestCase):
    def test_linear_fit(self):
        # fit a line to 2,2,2,2,2
        h = Hist1D(np.arange(10) + 0.5, bins="5,0,10")
        result = h.fit("a+b*x", draw=False)

        dpv = dict(zip(result["parnames"], result["parvalues"]))
        dpe = dict(zip(result["parnames"], result["parerrors"]))

        self.assertTrue(np.abs(dpv["a"] - 2) < 1e-3)
        self.assertTrue(np.abs(dpv["b"] - 0) < 1e-3)
        self.assertTrue(dpe["a"] > 0.5)

        self.assertTrue(result["chi2"] < 1e-3)
        self.assertEqual(result["ndof"], 3)

    @utils.ignore_division_errors
    def test_likelihood(self):
        h = Hist1D(np.arange(5) * 2, bins="10,0,10")
        f = "p0+0*x"
        ret_chi2 = h.fit(f, draw=False, likelihood=False)
        ret_like = h.fit(f, draw=False, likelihood=True)
        self.assertTrue(np.isclose(ret_chi2["parvalues"][0], 1.0))
        self.assertTrue(np.isclose(ret_like["parvalues"][0], 0.5))


class Hist1DTest(unittest.TestCase):
    pass

    # def __init__(self, obj=[], **kwargs):
    # def _copy(self):
    # def _init_numpy(self, obj, **kwargs):
    # def _extract_metadata(self, **kwargs):
    # def errors(self):
    # def errors_up(self):
    # def errors_down(self):
    # def counts(self):
    # def edges(self):
    # def bin_centers(self):
    # def bin_widths(self):
    # def nbins(self):
    # def integral(self):
    # def integral_error(self):
    # def mean(self):
    # def std(self):
    # def _fix_nan(self):
    # def __eq__(self, other):
    # def __ne__(self, other):
    # def __add__(self, other):
    # def __sub__(self, other):
    # def divide(self, other, binomial=False):
    # def __div__(self, other):
    # def __mul__(self, fact):
    # def __pow__(self, expo):
    # def normalize(self):
    # def rebin(self, nrebin):
    # def cumulative(self, from_left=True):
    #     from_left : bool, default True
    # def to_json(self):
    # def from_json(cls, obj):
    # def from_bincounts(cls, counts, bins, errors=None):


class Hist2DTest(unittest.TestCase):
    pass

    # def _init_numpy(self, obj, **kwargs):
    # def _check_consistency(self, other, raise_exception=True):
    # def __eq__(self, other):
    # def bin_centers(self):
    # def bin_widths(self):
    # def projection(self, axis):
    # def profile(self, axis):
    # def transpose(self):
    # def rebin(self, nrebinx, nrebiny=None):


if __name__ == "__main__":
    unittest.main()
