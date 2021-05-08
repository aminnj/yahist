import numpy as np

from yahist import Hist1D, Hist2D, utils
import os

np.set_printoptions(linewidth=120)

import pytest


def allclose(a, b, equal_nan=False):
    return np.testing.assert_allclose(np.array(a), np.array(b), equal_nan=equal_nan)


def test_integral():
    N = 100
    v = np.random.random(N)
    h = Hist1D(v, bins="10,0,1")
    assert h.integral == 100.0
    assert h.integral_error == 100.0 ** 0.5


def test_basic():
    v = np.array([0.5, 0.5, 1.5, 1.5])
    bins = np.array([0.0, 1.0, 2.0])
    h = Hist1D(v, bins=bins)
    assert h.dim == 1
    a = np.array([2.0, 2.0])
    allclose(h.counts, a)
    allclose(h.errors, a ** 0.5)
    allclose(h.edges, bins)
    assert h.nbins == len(bins) - 1
    allclose(h.bin_widths, np.array([1.0, 1.0]))
    allclose(h.bin_centers, np.array([0.5, 1.5]))


def test_from_list():
    h = Hist1D([-1, 1], weights=[1, 2])
    assert h.integral == 3.0


def test_weighted():
    v = np.array([0.5, 0.5, 1.5, 1.5])
    w = np.array([1.0, 1.0, 2.0, 2.0])
    bins = np.array([0.0, 1.0, 2.0])
    h = Hist1D(v, bins=bins, weights=w)
    allclose(h.counts, np.array([2.0, 4.0]))
    allclose(h.errors, np.array([2.0 ** 0.5, 8 ** 0.5]))


def test_weight_inputs():
    v = np.array([0.5, 0.5, 1.5, 1.5])
    h = Hist1D(v, weights=None)
    assert h.integral == 4


def test_nonuniform_binning():
    bins = np.array([0, 1, 10, 100, 1000])
    centers = np.array([0.5, 5.5, 55, 550])
    h = Hist1D(centers, bins=bins)
    allclose(h.counts, np.ones(len(centers)))
    allclose(h.bin_centers, centers)


def test_statistics():
    v = [0.5, 0.5, 1.5, 1.5]
    bins = [0.0, 1.0, 2.0]
    h = Hist1D(v, bins=bins)
    assert h.mean() == 1.0
    assert h.std() == 0.5
    assert h.mode() == 0.5

    v = [0.5, 1.5, 1.5]
    bins = [0.0, 1.0, 2.0]
    h = Hist1D(v, bins=bins)
    assert h.mode() == 1.5


def test_binning():
    v = np.arange(10)
    h1 = Hist1D(v, bins=np.linspace(0, 10, 11))
    h2 = Hist1D(v, bins="10,0,10")
    h3 = Hist1D(v, bins=10, range=[0, 10])
    assert h1 == h2
    assert h2 == h3


def test_integer_binning():
    v = np.arange(100).astype(int)
    edges = np.linspace(-0.5, 99.5, 101)
    h1 = Hist1D(v)
    allclose(h1.edges, edges)


def test_overflow():
    v = np.arange(10)
    bins = "8,0.5,8.5"
    h = Hist1D(v, bins=bins, overflow=True)
    assert h.counts[0] == 2
    assert h.counts[-1] == 2
    assert h.integral == 10

    h = Hist1D(v, bins=bins, overflow=False)
    assert h.counts[0] == 1
    assert h.counts[-1] == 1
    assert h.integral == 8


def test_median_quantiles():
    np.random.seed(42)
    v = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(1, 1, 100),])
    h = Hist1D(v, bins="100,-5,5")
    qs = np.linspace(0, 1, 11)
    v1 = np.quantile(v, qs)
    v2 = h.quantile(qs)
    bw = h.bin_widths[0]
    assert np.all(np.abs(v1 - v2) < bw)
    assert np.abs(np.median(v) - h.median()) < bw


def test_idempotence():
    h1 = Hist1D([0.5], bins=[0.0, 1])
    h2 = Hist1D(h1, label="test")
    assert h1 == h2


def test_sum():
    h1 = Hist1D([0.5], bins=[0.0, 1])
    h2 = Hist1D([0.5], bins=[0.0, 1])
    assert sum([h1, h2]) == (h1 + h2)

    h1 = Hist1D.from_bincounts([2])
    h2 = Hist1D.from_bincounts([3])
    h3 = Hist1D.from_bincounts([4])
    assert (h1 + h2 + h3).integral == 9
    assert sum([h1, h2, h3]).integral == 9


def test_metadata():
    assert Hist1D().metadata == {}
    assert Hist1D(label="test").metadata == {"label": "test"}
    assert Hist1D(color="C0").metadata == {"color": "C0"}
    assert Hist1D(color="C0", metadata={"foo": "bar"}).metadata == {
        "color": "C0",
        "foo": "bar",
    }
    assert Hist1D(metadata={"color": "C0", "foo": "bar"}).metadata == {
        "color": "C0",
        "foo": "bar",
    }


def test_copy():
    h1 = Hist1D([0.5], bins=[0.0, 1])
    h2 = h1.copy()
    assert h1 == h2
    h2._counts[0] *= 2.0
    assert h1 != h2


def test_arithmetic():
    def check_count_error(h, count, error):
        assert h.counts[0] == count
        assert h.errors[0] == error

    h = Hist1D([0.5], bins=[0.0, 1])

    check_count_error(h + h, 2.0, 2.0 ** 0.5)
    check_count_error(2.0 * h, 2.0, 2.0)
    check_count_error(h / 2.0, 0.5, 0.5)
    check_count_error(h - h, 0.0, 2.0 ** 0.5)
    check_count_error(h / h, 1.0, 2.0 ** 0.5)


def test_normalize():
    h1 = Hist1D([0.5, 1.5], bins=[0, 1, 2])
    h2 = h1.normalize()
    assert h2.integral == 1.0
    assert h2.integral_error == 0.5 ** 0.5


def test_density():
    v = np.random.normal(0, 1, 100)
    h = Hist1D(v, bins="5,-5,5", overflow=False).normalize(density=True)
    counts, edges = np.histogram(v, bins=h.edges, density=True)
    allclose(h.counts, counts)


def test_rebin():
    h1 = Hist1D([0.5, 1.5, 2.5, 3.5], bins=[0, 1, 2, 3, 4])
    h2 = h1.rebin(2)
    assert h1.integral == h2.integral
    assert h1.integral_error == h2.integral_error
    allclose(h2.edges, np.array([0.0, 2.0, 4.0]))


def test_restrict():
    h = Hist1D(np.arange(10), bins="10,0,10")
    assert h.restrict() == h
    assert h.restrict(None, None) == h
    assert h.restrict(None, 5).nbins == 5
    assert h.restrict(5, None).nbins == 5


def test_cumulative():
    h1 = Hist1D([0.5, 1.5, 2.5, 3.5], bins=[0, 1, 2, 3, 4])
    allclose(h1.cumulative(forward=True).counts, np.array([1, 2, 3, 4]))
    allclose(h1.cumulative(forward=False).counts, np.array([4, 3, 2, 1]))


def test_lookup():
    h = Hist1D.from_random(size=50, bins="7,-3,3", random_state=42)
    allclose(h.lookup(h.bin_centers), h.counts)
    assert h.lookup([-10.0]) == h.counts[0]
    assert h.lookup([10.0]) == h.counts[-1]
    assert h.lookup(-10.0) == h.counts[0]
    assert h.lookup(10.0) == h.counts[-1]


def test_sample():
    h1 = Hist1D.from_random("norm", bins="10,-5,5", size=1e4)
    h2 = Hist1D(h1.sample(size=1e5), bins=h1.edges)
    # fitting the ratio of the two should give a horizontal line at y=1
    ret = (h1.normalize() / h2.normalize()).fit("slope*x+offset")
    assert abs(ret["params"]["slope"]["value"]) < 0.05
    assert abs(ret["params"]["offset"]["value"] - 1) < 0.01


def test_gaussian():
    np.random.seed(42)
    for mean, sigma in [[0, 1], [1, 2]]:
        h1 = Hist1D.from_random(
            "norm", params=[mean, sigma], bins="10,-5,5", size=1e4, overflow=False
        )
        for likelihood in [True, False]:
            fit = h1.fit("gaus", likelihood=likelihood)
            assert abs(fit["params"]["mean"]["value"] - h1.mean()) < 0.2
            assert abs(fit["params"]["sigma"]["value"] - h1.std()) / h1.std() < 0.1
            assert (
                abs(fit["params"]["constant"]["value"] - h1.counts.max())
                / h1.counts.max()
                < 0.2
            )


def test_json():
    h1 = Hist1D([0.5], bins=[0.0, 1], label="foo")

    h2 = h1.from_json(h1.to_json())
    assert h1 == h2
    assert h1.metadata == h2.metadata

    h1.to_json(".tmphist.json")
    h2 = h1.from_json(".tmphist.json")
    assert h1 == h2
    assert h1.metadata == h2.metadata


def test_frombincounts():
    np.random.seed(42)
    v = np.random.random(100)
    bins = np.linspace(0, 1, 11)
    h1 = Hist1D(v, bins=bins)
    counts, _ = np.histogram(v, bins=bins)
    h2 = Hist1D.from_bincounts(counts=counts, bins=bins)
    assert h1 == h2
    h3 = Hist1D.from_bincounts([1, 2])
    assert h3.nbins == 2
    assert h3.integral == 3.0

    h = Hist1D.from_bincounts(
        [1, 1, 2], [-1.5, -0.5, 0.5, 1.5], label="test1", color="red"
    )
    allclose(h.counts, [1.0, 1.0, 2.0])
    assert h.metadata["label"] == "test1"


def test_fromrandom():
    h = Hist1D.from_random("norm", params=[0, 1], size=1e3, random_state=42)
    assert abs(h.mean()) < 0.1
    assert 0.9 < h.std() < 1.1


def test_datetime():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame()
    df["date"] = pd.date_range("2019-01-01", "2020-01-10", freq="1h")
    df["num"] = np.random.normal(0, 1, len(df))

    bins = pd.date_range(
        pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-10"), periods=20
    )
    h1 = Hist1D(df["date"])
    h2 = Hist1D(df["date"], bins=10)
    h3 = Hist1D(df["date"], bins=bins)

    for h in [h1, h2, h3]:
        assert len(df) == h.integral


def test_fill():
    h = Hist1D(bins="10,0,10", label="test")
    h.fill([1, 2, 3, 4])
    h.fill([0, 1, 2])
    h.median()
    assert h.lookup(0) == 1.0
    assert h.lookup(1) == 2.0
    assert h.lookup(3) == 1.0
    assert h.lookup(5) == 0.0
    assert h.metadata["label"] == "test"


if __name__ == "__main__":
    pytest.main(["--capture=no", __file__])
