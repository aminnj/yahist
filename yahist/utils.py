from __future__ import print_function

import matplotlib
import numpy as np


def is_listlike(obj):
    return hasattr(obj, "__array__") or type(obj) in [list, tuple]

def has_uniform_spacing(obj, epsilon=1e-6):
    offsets = np.ediff1d(obj)
    return np.all(offsets-offsets[0] < epsilon)

def set_default_style():
    from matplotlib import rcParams

    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [
        "Helvetica",
        "Arial",
        "Liberation Sans",
        "Bitstream Vera Sans",
        "DejaVu Sans",
    ]
    rcParams["legend.fontsize"] = 11
    rcParams["legend.labelspacing"] = 0.2
    rcParams["hatch.linewidth"] = 0.5
    rcParams["axes.xmargin"] = 0.0  # rootlike, no extra padding within x axis
    rcParams["axes.labelsize"] = "x-large"
    rcParams["axes.formatter.use_mathtext"] = True
    rcParams["legend.framealpha"] = 0.65
    rcParams["axes.labelsize"] = "x-large"
    rcParams["axes.titlesize"] = "large"
    rcParams["xtick.labelsize"] = "large"
    rcParams["ytick.labelsize"] = "large"
    rcParams["figure.subplot.hspace"] = 0.1
    rcParams["figure.subplot.wspace"] = 0.1
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.max_open_warning"] = 0
    rcParams["figure.dpi"] = 100
    rcParams["axes.formatter.limits"] = [-5, 4]


def compute_darkness(r, g, b, a=1.0):
    # darkness = 1 - luminance
    return a * (1.0 - (0.299 * r + 0.587 * g + 0.114 * b))


def clopper_pearson_error(passed, total, level=0.6827):
    """
    matching TEfficiency::ClopperPearson()
    """
    import scipy.stats

    alpha = 0.5 * (1.0 - level)
    low = scipy.stats.beta.ppf(alpha, passed, total - passed + 1)
    high = scipy.stats.beta.ppf(1 - alpha, passed + 1, total - passed)
    return low, high


def poisson_errors(obs, alpha=1 - 0.6827):
    """
    Return poisson low and high values for a series of data observations
    """
    from scipy.stats import gamma

    lows = np.nan_to_num(gamma.ppf(alpha / 2, np.array(obs)))
    highs = np.nan_to_num(gamma.ppf(1.0 - alpha / 2, np.array(obs) + 1))
    return lows, highs


def binomial_obs_z(data, bkg, bkgerr, gaussian_fallback=True):
    """
    Calculate pull values according to
    https://root.cern.ch/doc/v606/NumberCountingUtils_8cxx_source.html#l00137
    The scipy version is vectorized, so you can feed in arrays
    If `gaussian_fallback` return a simple gaussian pull when data count is 0,
    otherwise both ROOT and scipy will return inf/nan.
    """
    from scipy.special import betainc
    import scipy.stats as st

    z = np.ones(len(data))
    nonzeros = data > 1.0e-6
    tau = 1.0 / bkg[nonzeros] / (bkgerr[nonzeros] / bkg[nonzeros]) ** 2.0
    auxinf = bkg[nonzeros] * tau
    v = betainc(data[nonzeros], auxinf + 1, 1.0 / (1.0 + tau))
    z[nonzeros] = st.norm.ppf(1 - v)
    if (data < 1.0e-6).sum():
        zeros = data < 1.0e-6
        z[zeros] = -(bkg[zeros]) / np.hypot(
            poisson_errors(data[zeros])[1], bkgerr[zeros]
        )
    return z


def nan_to_num(f):
    def g(*args, **kw):
        return np.nan_to_num(f(*args, **kw))

    return g


def ignore_division_errors(f):
    def g(*args, **kw):
        with np.errstate(divide="ignore", invalid="ignore"):
            return f(*args, **kw)

    return g
