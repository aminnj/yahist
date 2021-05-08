import numpy as np

from yahist import Hist1D, Hist2D, utils
import os

from yahist.fit import expr_to_lambda

np.set_printoptions(linewidth=120)

import pytest


def allclose(a, b, equal_nan=False):
    return np.testing.assert_allclose(np.array(a), np.array(b), equal_nan=equal_nan)


def test_linear_fit():
    # fit a line to 2,2,2,2,2
    h = Hist1D(np.arange(10) + 0.5, bins="5,0,10")
    result = h.fit("a+b*x", draw=False)

    assert np.abs(result["params"]["a"]["value"] - 2) < 1e-3
    assert np.abs(result["params"]["b"]["value"] - 0) < 1e-3
    assert result["params"]["a"]["error"] > 0.5

    assert result["chi2"] < 1e-3
    assert result["ndof"] == 3


def test_extent():
    h_full = Hist1D.from_random(
        "uniform", params=[-1, 1], size=1e3, bins="100,-1,1"
    ) + Hist1D.from_random("uniform", params=[0, +1], size=2e3, bins="100,-1,1")

    h_restricted = h_full.restrict(0, 1)
    fit_full = h_full.fit("a+b*x", extent=[0, 1], color="C0", draw=False)
    fit_restricted = h_restricted.fit("a+b*x", color="C3", draw=False)

    # fitted values should be the same since the domain is the same
    assert fit_full["params"] == fit_restricted["params"]

    # check that the histograms of the fit match the input domain
    assert h_restricted._check_consistency(fit_restricted["hfit"])
    assert h_full._check_consistency(fit_full["hfit"])


@utils.ignore_division_errors
def test_likelihood():
    h = Hist1D(np.arange(5) * 2, bins="10,0,10")
    f = "p0+0*x"
    ret_chi2 = h.fit(f, draw=False, likelihood=False)
    ret_like = h.fit(f, draw=False, likelihood=True)
    allclose(ret_chi2["params"]["p0"]["value"], 1.0)
    allclose(ret_like["params"]["p0"]["value"], 0.5)

    # all relative errors within <1% when counts are large
    h = Hist1D.from_random(
        "norm", params=[0, 1], size=2000, random_state=42, bins="20,-3,3"
    )
    ret_chi2 = h.fit(
        "a*np.exp(-(x-mu)**2./(2*sigma**2.))", draw=False, likelihood=False
    )
    ret_like = h.fit("a*np.exp(-(x-mu)**2./(2*sigma**2.))", draw=False, likelihood=True)
    keys = ret_chi2["params"].keys()
    ret_chi2_errors = np.array([ret_chi2["params"][key]["error"] for key in keys])
    ret_like_errors = np.array([ret_like["params"][key]["error"] for key in keys])
    ret_chi2_values = np.array([ret_chi2["params"][key]["value"] for key in keys])
    v = (ret_chi2_errors - ret_like_errors) / ret_chi2_values
    assert (np.abs(v) < 0.01).mean() == 1.0


def test_against_root():
    """
    import ROOT as r

    h1 = r.TH1F("h1","",10,-1.01,1.01)
    for x in [-0.1, 0.1, -0.2, 0.1, 0.1, 0.4, 0.1]:
        h1.Fill(x)
    for likelihood in ["", "L"]:
        res = h1.Fit("gaus", f"QS{likelihood}").Get()
        print(list(res.Parameters()), list(res.Errors()))
    """
    h = Hist1D([-0.1, 0.1, -0.2, 0.1, 0.1, 0.4, 0.1], bins="10,-1.01,1.01")

    res = h.fit("gaus", likelihood=False)
    params = res["params"]
    assert abs(params["constant"]["value"] - 4.1175) < 1e-3
    assert abs(params["mean"]["value"] - 0.0673) < 1e-3
    assert abs(params["sigma"]["value"] - 0.1401) < 1e-3
    assert abs(params["constant"]["error"] - 2.0420) < 1e-3
    assert abs(params["mean"]["error"] - 0.0584) < 1e-3
    assert abs(params["sigma"]["error"] - 0.0531) < 1e-3

    res = h.fit("gaus", likelihood=True)
    params = res["params"]
    assert abs(params["constant"]["value"] - 4.3562) < 1e-3
    assert abs(params["mean"]["value"] - 0.07190) < 1e-3
    assert abs(params["sigma"]["value"] - 0.1294) < 1e-3
    assert abs(params["constant"]["error"] - 2.0008) < 2e-2
    assert abs(params["mean"]["error"] - 0.04908) < 1e-3
    assert abs(params["sigma"]["error"] - 0.0339) < 1e-3


def test_gaus_extra():
    np.random.seed(42)
    bins = "50,-5,5"
    mean = 1.0
    sigma = 0.5
    h = Hist1D(np.random.normal(mean, sigma, 350), bins=bins) + Hist1D(
        10 * np.random.random(600) - 5, bins=bins
    )
    params = h.fit("offset+gaus", draw=False)["params"]
    assert abs(params["mean"]["value"] - mean) / mean < 0.1
    assert abs(params["sigma"]["value"] - sigma) / sigma < 0.2


def test_expr_to_lambda():
    f = expr_to_lambda("x+a+a+b+np.pi")
    g = lambda x, a, b: x + a + a + b + np.pi
    assert f(1, 2, 3) == g(1, 2, 3)

    f = expr_to_lambda("m*x+b")
    g = lambda x, m, b: m * x + b
    assert f(1, 2, 3) == g(1, 2, 3)

    f = expr_to_lambda("1+np.poly1d([a,b,c])(x)")
    g = lambda x, a, b, c: 1 + np.poly1d([a, b, c])(x)
    assert f(1, 2, 3, 4) == g(1, 2, 3, 4)

    f = expr_to_lambda("const + norm*np.exp(-(x-mu)**2/(2*sigma**2))")
    g = lambda x, const, norm, mu, sigma: const + norm * np.exp(
        -((x - mu) ** 2) / (2 * sigma ** 2)
    )
    assert f(1, 2, 3, 4, 5) == g(1, 2, 3, 4, 5)


if __name__ == "__main__":
    pytest.main(["--capture=no", __file__])
