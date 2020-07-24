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

        # all relative errors within <1% when counts are large
        h = Hist1D.from_random("norm",params=[0,1], size=2000, random_state=42, bins="20,-3,3")
        ret_chi2 = h.fit("a*np.exp(-(x-mu)**2./(2*sigma**2.))", draw=False, likelihood=False)
        ret_like = h.fit("a*np.exp(-(x-mu)**2./(2*sigma**2.))", draw=False, likelihood=True)
        v = (ret_chi2["parerrors"] - ret_like["parerrors"])/ret_chi2["parvalues"]
        self.assertEqual((np.abs(v) < 0.01).mean(), 1.)


class Hist1DTest(unittest.TestCase):
    def test_integral(self):
        N = 100
        v = np.random.random(N)
        h = Hist1D(v, bins="10,0,1")
        self.assertEqual(h.integral, 100.0)
        self.assertEqual(h.integral_error, 100.0 ** 0.5)

    def test_basic(self):
        v = np.array([0.5, 0.5, 1.5, 1.5])
        bins = np.array([0.0, 1.0, 2.0])
        h = Hist1D(v, bins=bins)
        a = np.array([2.0, 2.0])
        self.assertTrue(np.allclose(h.counts, a))
        self.assertTrue(np.allclose(h.errors, a ** 0.5))
        self.assertTrue(np.allclose(h.edges, bins))
        self.assertEqual(h.nbins, len(bins) - 1)
        self.assertTrue(np.allclose(h.bin_widths, np.array([1.0, 1.0])))
        self.assertTrue(np.allclose(h.bin_centers, np.array([0.5, 1.5])))

    def test_weighted(self):
        v = np.array([0.5, 0.5, 1.5, 1.5])
        w = np.array([1.0, 1.0, 2.0, 2.0])
        bins = np.array([0.0, 1.0, 2.0])
        h = Hist1D(v, bins=bins, weights=w)
        self.assertTrue(np.allclose(h.counts, np.array([2.0, 4.0])))
        self.assertTrue(np.allclose(h.errors, np.array([2.0 ** 0.5, 8 ** 0.5])))

    def test_nonuniform_binning(self):
        bins = np.array([0, 1, 10, 100, 1000])
        centers = np.array([0.5, 5.5, 55, 550])
        h = Hist1D(centers, bins=bins)
        self.assertTrue(np.allclose(h.counts, np.ones(len(centers))))
        self.assertTrue(np.allclose(h.bin_centers, centers))

    def test_statistics(self):
        v = [0.5, 0.5, 1.5, 1.5]
        bins = np.array([0.0, 1.0, 2.0])
        h = Hist1D(v, bins=bins)
        self.assertEqual(h.mean(), 1.0)
        self.assertEqual(h.std(), 0.5)

    def test_binning(self):
        v = np.arange(10)
        h1 = Hist1D(v, bins=np.linspace(0, 10, 11))
        h2 = Hist1D(v, bins="10,0,10")
        h3 = Hist1D(v, bins=10, range=[0, 10])
        self.assertEqual(h1, h2)
        self.assertEqual(h2, h3)

    def test_overflow(self):
        v = np.arange(10)
        bins = "8,0.5,8.5"
        h = Hist1D(v, bins=bins, overflow=True)
        self.assertEqual(h.counts[0], 2)
        self.assertEqual(h.counts[-1], 2)
        self.assertEqual(h.integral, 10)

        h = Hist1D(v, bins=bins, overflow=False)
        self.assertEqual(h.counts[0], 1)
        self.assertEqual(h.counts[-1], 1)
        self.assertEqual(h.integral, 8)

    def test_idempotence(self):
        h1 = Hist1D([0.5], bins=[0.0, 1])
        h2 = Hist1D(h1, label="test")
        self.assertEqual(h1, h2)

    def test_metadata(self):
        self.assertEqual(Hist1D().metadata, {})
        self.assertEqual(Hist1D(label="test").metadata, {"label": "test"})
        self.assertEqual(Hist1D(color="C0").metadata, {"color": "C0"})
        self.assertEqual(
            Hist1D(color="C0", metadata={"foo": "bar"}).metadata,
            {"color": "C0", "foo": "bar"},
        )
        self.assertEqual(
            Hist1D(metadata={"color": "C0", "foo": "bar"}).metadata,
            {"color": "C0", "foo": "bar"},
        )

    def test_copy(self):
        h1 = Hist1D([0.5], bins=[0.0, 1])
        h2 = h1.copy()
        self.assertEqual(h1, h2)
        h2._counts[0] *= 2.0
        self.assertNotEqual(h1, h2)

    def test_arithmetic(self):
        def check_count_error(h, count, error):
            self.assertEqual(h.counts[0], count)
            self.assertEqual(h.errors[0], error)

        h = Hist1D([0.5], bins=[0.0, 1])

        check_count_error(h + h, 2.0, 2.0 ** 0.5)
        check_count_error(2.0 * h, 2.0, 2.0)
        check_count_error(h / 2.0, 0.5, 0.5)
        check_count_error(h - h, 0.0, 2.0 ** 0.5)
        check_count_error(h / h, 1.0, 2.0 ** 0.5)

    def test_normalize(self):
        h1 = Hist1D([0.5, 1.5], bins=[0, 1, 2])
        h2 = h1.normalize()
        self.assertEqual(h2.integral, 1.0)
        self.assertEqual(h2.integral_error, 0.5 ** 0.5)

    def test_rebin(self):
        h1 = Hist1D([0.5, 1.5, 2.5, 3.5], bins=[0, 1, 2, 3, 4])
        h2 = h1.rebin(2)
        self.assertEqual(h1.integral, h2.integral)
        self.assertEqual(h1.integral_error, h2.integral_error)
        self.assertTrue(np.allclose(h2.edges, np.array([0.0, 2.0, 4.0])))

    def test_cumulative(self):
        h1 = Hist1D([0.5, 1.5, 2.5, 3.5], bins=[0, 1, 2, 3, 4])
        self.assertTrue(
            np.allclose(h1.cumulative(forward=True).counts, np.array([1, 2, 3, 4]))
        )
        self.assertTrue(
            np.allclose(h1.cumulative(forward=False).counts, np.array([4, 3, 2, 1]))
        )

    def test_json(self):
        h1 = Hist1D([0.5], bins=[0.0, 1], label="foo")

        h2 = h1.from_json(h1.to_json())
        self.assertEqual(h1, h2)
        self.assertEqual(h1.metadata, h2.metadata)

        h1.to_json(".tmphist.json")
        h2 = h1.from_json(".tmphist.json")
        self.assertEqual(h1, h2)
        self.assertEqual(h1.metadata, h2.metadata)

    def test_frombincounts(self):
        np.random.seed(42)
        v = np.random.random(100)
        bins = np.linspace(0, 1, 11)
        h1 = Hist1D(v, bins=bins)
        counts, _ = np.histogram(v, bins=bins)
        h2 = Hist1D.from_bincounts(counts=counts, bins=bins)
        self.assertEqual(h1, h2)

    def test_fromrandom(self):
        h = Hist1D.from_random("norm", params=[0, 1], size=1e3, random_state=42)
        self.assertTrue(abs(h.mean()) < 0.1)
        self.assertTrue(0.9 < h.std() < 1.1)


class Hist2DTest(unittest.TestCase):
    def test_basic(self):
        xs = np.array([0.5, 1.5])
        ys = np.array([1.5, 0.5])
        bins = np.array([0, 1, 2])
        h1 = Hist2D(np.c_[xs, ys], bins=bins)
        counts, edgesx, edgesy = np.histogram2d(xs, ys, bins)
        h2 = Hist2D.from_bincounts(counts, (edgesx, edgesy))
        self.assertEqual(h1, h2)
        self.assertEqual(h1.nbins, (2, 2))
        self.assertEqual(h1.integral, 2.0)
        self.assertEqual(h1.integral_error, 2.0 ** 0.5)

        self.assertTrue(np.allclose(h1.edges[0], edgesx))
        self.assertTrue(np.allclose(h1.edges[1], edgesy))

        self.assertEqual(h1, h1.transpose())

    def test_reductions(self):
        xs = np.array([0.5, 1.5])
        ys = np.array([1.5, 0.5])
        bins = np.array([0, 1, 2])
        h1 = Hist2D(np.c_[xs, ys], bins=bins)
        self.assertEqual(h1, h1.rebin(1))
        self.assertEqual(h1, h1.rebin(1, 1))

        h2 = h1.rebin(2)
        self.assertEqual(h2.nbins, (1, 1))

        self.assertEqual(h1.projection("x"), h1.projection("y"))
        self.assertEqual(h1.profile("x"), h1.profile("y"))

        self.assertTrue(np.allclose(h1.projection("x").counts, np.array([1.0, 1.0])))
        self.assertTrue(np.allclose(h1.profile("x").counts, np.array([1.5, 0.5])))

    def test_fromrandom(self):
        mus = [0, 0]
        cov = [[1, 0], [0, 1]]
        h = Hist2D.from_random(
            "multivariate_normal", params=[mus, cov], size=1e4, random_state=42
        )
        for axis in ["x", "y"]:
            self.assertTrue(abs(h.projection(axis).mean()) < 0.1)
            self.assertTrue(0.9 < h.projection(axis).std() < 1.1)


if __name__ == "__main__":
    unittest.main()
