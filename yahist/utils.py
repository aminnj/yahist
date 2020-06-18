from __future__ import print_function

import matplotlib
import numpy as np


def is_listlike(obj):
    return hasattr(obj, "__array__") or type(obj) in [list, tuple]


def has_uniform_spacing(obj, epsilon=1e-6):
    offsets = np.ediff1d(obj)
    return np.all(offsets - offsets[0] < epsilon)


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


def fit_hist(func, hist, nsamples=500, ax=None, draw=True, color="red"):
    """
    Fits a function to a histogram via `scipy.optimize.curve_fit`,
    calculating a 1-sigma band, and optionally plotting it.
    Note that this does not support asymmetric errors. It will
    symmetrize such errors prior to fitting. Empty bins are excluded
    from the fit.

    Parameters
    ----------
    func : function taking x data as the first argument, followed by parameters
    hist : Hist1D
    nsamples : number of samples/bootstraps for calculating error bands
    ax : matplotlib AxesSubplot object, default None
    draw : bool, default True
       draw to a specified or pre-existing AxesSubplot object
    color : str, default "red"
       color of fit line and error band

    Returns
    -------
    dict of
        - x data, y data, y errors, fit y values, fit y errors
        - parameter names/values and covariances as returned by `scipy.optimize.curve_fit`
        - parameter errors (sqrt of diagonal elements of the covariance matrix)
        - a Hist1D object containing the fit

    Example
    -------
    >>> h = Hist1D(np.random.random(1000), bins="30,0,1.5")
    >>> h.plot(show_errors=True, color="k")
    >>> res = fit_hist(lambda x,a,b: a+b*x, h)
    >>> print(res["parnames"],res["parvalues"],res["parerrors"])
    """
    from scipy.optimize import curve_fit
    from . import Hist1D

    if draw and not ax:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    xdataraw = hist.bin_centers
    ydataraw = hist.counts
    yerrsraw = hist.errors

    tomask = (ydataraw == 0.0) & (yerrsraw == 0.0)
    xdata = xdataraw[~tomask]
    ydata = ydataraw[~tomask]
    yerrs = yerrsraw[~tomask]

    popt, pcov = curve_fit(func, xdata, ydata, sigma=yerrs, absolute_sigma=True)

    vopts = np.random.multivariate_normal(popt, pcov, nsamples)
    sampled_ydata = np.vstack([func(xdataraw, *vopt).T for vopt in vopts])
    sampled_means = sampled_ydata.mean(axis=0)
    sampled_stds = sampled_ydata.std(axis=0)

    fit_ydata = func(xdataraw, *popt)

    if draw:
        ax.plot(xdataraw, fit_ydata, color=color)
        ax.fill_between(
            xdataraw,
            fit_ydata - sampled_stds,
            fit_ydata + sampled_stds,
            facecolor=color,
            alpha=0.15,
            label=r"fit $\pm$1$\sigma$",
        )

    hfit = Hist1D.from_bincounts(fit_ydata, hist.edges, errors=sampled_stds)

    return dict(
        xdata=xdataraw,
        ydata=ydataraw,
        yerrs=yerrsraw,
        yfit=fit_ydata,
        yfiterrs=sampled_stds,
        parnames=func.__code__.co_varnames[1:],
        parvalues=popt,
        parerrors=np.diag(pcov) ** 0.5,
        pcov=pcov,
        hfit=hfit,
    )
