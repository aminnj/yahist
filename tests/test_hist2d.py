import numpy as np

from yahist import Hist1D, Hist2D, utils
import os

np.set_printoptions(linewidth=120)

import pytest


def allclose(a, b, equal_nan=False):
    return np.testing.assert_allclose(np.array(a), np.array(b), equal_nan=equal_nan)


def test_basic():
    xs = np.array([0.5, 1.5])
    ys = np.array([1.5, 0.5])
    bins = np.array([0, 1, 2])
    h1 = Hist2D(np.c_[xs, ys], bins=bins)
    counts, edgesx, edgesy = np.histogram2d(xs, ys, bins)
    h2 = Hist2D.from_bincounts(counts, (edgesx, edgesy))
    assert h1 == h2
    assert h1.nbins == (2, 2)
    assert h1.integral == 2.0
    assert h1.integral_error == 2.0 ** 0.5

    allclose(h1.edges[0], edgesx)
    allclose(h1.edges[1], edgesy)

    assert h1 == h1.transpose()


def test_tuple_input():
    xs = np.array([0.5, 1.5])
    ys = np.array([1.5, 0.5])
    bins = np.array([0, 1, 2])
    h1 = Hist2D((xs, ys), bins=bins)
    counts, edgesx, edgesy = np.histogram2d(xs, ys, bins)
    h2 = Hist2D.from_bincounts(counts, (edgesx, edgesy))
    assert h1 == h2


def test_equality():
    v = np.random.normal(0, 1, size=(1000, 2))
    h1 = Hist2D(v, bins=np.linspace(-5, 5, 11))
    h2 = Hist2D(v, bins=[np.linspace(-5, 5, 11), np.linspace(-5, 5, 11)])
    h3 = Hist2D(v, bins="10,-5,5")
    h4 = Hist2D(v, bins="10,-5,5,10,-5,5")
    assert h1 == h2
    assert h1 == h3
    assert h1 == h4

    h1 = Hist2D(v, bins=[np.linspace(-5, 5, 11), np.linspace(-8, 8, 11)])
    h2 = Hist2D(v, bins="10,-5,5,10,-8,8")
    assert h1 == h2


def test_weight_inputs():
    v = np.array([0.5, 0.5, 1.5, 1.5])
    h = Hist2D(np.c_[v, v], weights=None)
    assert h.integral == 4


def test_profile():
    xs = np.array([0.5, 1.5])
    ys = np.array([1.5, 0.5])
    bins = np.array([0, 1, 2])
    h1 = Hist2D(np.c_[xs, ys], bins=bins)
    assert h1.profile("x") == h1.profile("y")
    allclose(h1.profile("x").counts, np.array([1.5, 0.5]))


def test_projection():
    xs = np.array([0.5, 1.5])
    ys = np.array([1.5, 0.5])
    bins = np.array([0, 1, 2])
    h1 = Hist2D(np.c_[xs, ys], bins=bins)
    assert h1.projection("x") == h1.projection("y")
    allclose(h1.projection("x").counts, np.array([1.0, 1.0]))


def test_sum():
    h1 = Hist2D.from_random(size=100, bins="5,0,5")
    h2 = Hist2D.from_random(size=100, bins="5,0,5")
    assert sum([h1, h2]) == (h1 + h2)
    assert (Hist2D() + h1) == h1


def test_rebin():
    xs = np.array([0.5, 1.5])
    ys = np.array([1.5, 0.5])
    bins = np.array([0, 1, 2])
    h1 = Hist2D(np.c_[xs, ys], bins=bins)
    assert h1 == h1.rebin(1)
    assert h1 == h1.rebin(1, 1)

    h2 = h1.rebin(2)
    assert h2.nbins == (1, 1)

    bins = [np.array([0, 1, 2]), np.array([0, 1, 2, 3])]
    h1 = Hist2D(np.c_[xs, ys], bins=bins)
    assert h1.nbins == (2, 3)
    assert h1.rebin(2, 3).nbins == (1, 1)


def test_restrict():
    xs = [0, 1, 2, 2, 2]
    ys = [1, 2, 1, 1, 1]
    h = Hist2D(np.c_[xs, ys], bins="6,0.5,3.5")
    h = h.restrict(2.0, None, None, 1.5)
    assert h.integral == 3.0
    assert h.nbins == (3, 2)


def test_smooth():
    h = Hist2D.from_random(bins="5,0,5")
    assert h == h.smooth(window=1, ntimes=1)
    assert h == h.smooth(window=1, ntimes=10)
    m = np.array([[2, 2],])
    h = Hist2D(m, bins="5,-0.5,4.5")
    h = h.smooth(window=3, ntimes=1)
    assert h.lookup(2 + 1, 2) == h.lookup(2 - 1, 2)
    assert h.lookup(2, 2 + 1) == h.lookup(2, 2 - 1)


def test_sample():
    h1 = Hist2D.from_random(size=1e5)
    h2 = Hist2D(h1.sample(1e5), bins=h1.edges)
    result = (h1.projection() / h2.projection()).fit("a+b*x")
    slope = result["params"]["b"]
    assert abs(slope["error"]) > abs(slope["value"])


def test_cumulative():
    h = Hist2D.from_random(bins="5,0,5")
    assert h == h.cumulative(forwardx=None, forwardy=None)


def test_cumulativelookup():
    h = Hist2D.from_random(bins="5,0,5")
    assert h.cumulative().counts.max() == h.integral
    assert h.cumulative(forwardx=True, forwardy=True).lookup(4.9, 4.9) == h.integral
    assert h.cumulative(forwardx=False, forwardy=False).lookup(0.1, 0.1) == h.integral
    assert h.cumulative(forwardx=True, forwardy=False).lookup(4.9, 0.1) == h.integral
    assert h.cumulative(forwardx=False, forwardy=True).lookup(0.1, 4.9) == h.integral


def test_fromrandom():
    mus = [0, 0]
    cov = [[1, 0], [0, 1]]
    h = Hist2D.from_random(
        "multivariate_normal", params=[mus, cov], size=1e4, random_state=42
    )
    for axis in ["x", "y"]:
        assert abs(h.projection(axis).mean()) < 0.1
        assert 0.9 < h.projection(axis).std() < 1.1

    h = Hist2D.from_random("norm", params=[(2, 2)], bins=50)
    assert h.rebin(5).projection("x") == h.projection("x").rebin(5)


def test_threads():
    N = int(1e5) + 1
    x = np.random.normal(0, 1, N)
    y = np.random.normal(0, 1, N)
    xy = np.c_[x, y]
    bins = [np.linspace(-3, 3, 51), np.linspace(-3, 3, 51)]
    for overflow in [True, False]:
        h1 = Hist2D(xy, bins=bins, overflow=overflow)
        for threads in [None, 0, 1, 2]:
            h2 = Hist2D(xy, bins=bins, threads=threads, overflow=overflow)
            assert h1 == h2


if __name__ == "__main__":
    pytest.main(["--capture=no", __file__])
