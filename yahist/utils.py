from __future__ import print_function

import matplotlib
import numpy as np


def is_listlike(obj):
    return (type(obj) in [tuple, list]) or (np.ndim(obj) >= 1)


def is_datelike(obj):
    if hasattr(obj, "dtype"):
        # return isinstance(obj.dtype.type, (type(np.datetime64),))
        return obj.dtype.type == np.datetime64
    else:
        return False


def convert_dates(obj):
    import matplotlib.dates

    return matplotlib.dates.date2num(obj)


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


def binomial_obs_z(data, bkg, bkgerr):
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


def expr_to_lambda(expr):
    """
    Converts a string expression like
        "a+b*np.exp(-c*x+math.pi)"
    into a lambda function with 1 variable and N parameters,
        lambda x,a,b,c: "a+b*np.exp(-c*x+math.pi)"
    `x` is assumed to be the main variable.
    Very simple logic that ignores things like `foo.bar`
    or `foo(` from being considered a parameter.
    """
    from io import BytesIO
    from tokenize import tokenize, NAME

    varnames = []
    g = list(tokenize(BytesIO(expr.encode("utf-8")).readline))
    for ix, x in enumerate(g):
        toknum = x[0]
        tokval = x[1]
        if toknum != NAME:
            continue
        if ix > 0 and g[ix - 1][1] in ["."]:
            continue
        if ix < len(g) - 1 and g[ix + 1][1] in [".", "("]:
            continue
        varnames.append(tokval)
    varnames = [name for name in varnames if name != "x"]
    varnames = list(
        dict.fromkeys(varnames)
    )  # remove duplicates, preserving order (python>=3.7)
    lambdastr = f"lambda x,{','.join(varnames)}: {expr}"
    return eval(lambdastr)


def calculate_hessian(func, x0, epsilon=1.0e-5):
    """
    # Taken almost verbatim from https://gist.github.com/jgomezdans/3144636
    A numerical approximation to the Hessian matrix of cost function at
    location x0
    """
    from scipy.optimize import approx_fprime

    f1 = approx_fprime(x0, func, epsilon)
    n = len(x0)
    hess = np.zeros((n, n))
    xx = x0
    for j in range(n):
        xx0 = xx[j]  # Store old value
        xx[j] = xx0 + epsilon  # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = approx_fprime(x0, func, epsilon)
        hess[:, j] = (f2 - f1) / epsilon  # scale...
        xx[j] = xx0  # Restore initial value of x0
    return hess


def curve_fit_wrapper(
    func, xdata, ydata, sigma=None, absolute_sigma=True, likelihood=False, **kwargs
):
    """
    Wrapper around `scipy.optimize.curve_fit`. Initial parameters (`p0`)
    can be set in the function definition with defaults for kwargs
    (e.g., `func = lambda x,a=1.,b=2.: x+a+b`, will feed `p0 = [1.,2.]` to `curve_fit`)
    """
    from scipy.optimize import minimize, curve_fit

    xdata = xdata.astype(np.float64)  # need this for minuit hesse to converge
    if func.__defaults__ and len(func.__defaults__) + 1 == func.__code__.co_argcount:
        if "p0" not in kwargs:
            kwargs["p0"] = func.__defaults__
    tomask = (ydata == 0.0) | (~np.isfinite(ydata))
    if sigma is not None:
        tomask |= sigma == 0.0
    popt, pcov = curve_fit(
        func,
        xdata[~tomask],
        ydata[~tomask],
        sigma=sigma[~tomask],
        absolute_sigma=absolute_sigma,
        **kwargs,
    )
    if likelihood:
        from scipy.special import gammaln

        def fnll(v):
            ypred = func(xdata, *v)
            if (ypred <= 0.0).any():
                return 1e6
            return (
                ypred.sum() - (ydata * np.log(ypred)).sum() + gammaln(ydata + 1).sum()
            )

        res = minimize(fnll, popt, method="BFGS")
        popt = res.x
        pcov = res.hess_inv
        use_minuit = True
        if use_minuit:
            try:
                from iminuit import Minuit
            except ImportError:
                raise Exception(
                    "For likelihood minimization, the 'iminuit' module must be installed (`pip install --user iminuit`)."
                )
            m = Minuit(fnll, popt)
            m.errordef = 0.5
            m.hesse()
            pcov = np.array(m.covariance)
    return popt, pcov


def fit_hist(
    func,
    hist,
    nsamples=500,
    ax=None,
    draw=True,
    color="red",
    legend=True,
    label=r"fit $\pm$1$\sigma$",
    band_style="filled",
    likelihood=False,
    curve_fit_kwargs=dict(),
):
    r"""
    Fits a function to a histogram via `scipy.optimize.curve_fit`,
    calculating a 1-sigma band, and optionally plotting it.
    Note that this does not support asymmetric errors. It will
    symmetrize such errors prior to fitting. Empty bins are excluded
    from the fit.

    Parameters
    ----------
    func : function taking x data as the first argument, followed by parameters, or a string
    hist : Hist1D
    nsamples : int, default 500
        number of samples/bootstraps for calculating error bands
    ax : matplotlib AxesSubplot object, default None
    band_style : None/str, default None
        if not None, compute and display uncertainty band. Possible strings are
        "filled", "dashed", "dotted", "dashdot", "solid"
    draw : bool, default True
       draw to a specified or pre-existing AxesSubplot object
    color : str, default "red"
       color of fit line and error band
    curve_fit_kwargs : dict
       dict of extra kwargs to pass to `scipy.optimize.curve_fit`
    label : str, default r"fit $\pm$1$\sigma$"
       legend entry label. Parameters will be appended unless this
       is empty.
    legend : bool, default True
        if True and the histogram has a label, draw the legend

    Returns
    -------
    dict of
        - parameter names, values, errors (sqrt of diagonal of the cov. matrix)
        - chi2, ndof of fit
        - a Hist1D object containing the fit

    Example
    -------
    >>> h = Hist1D(np.random.random(1000), bins="30,0,1.5")
    >>> h.plot(show_errors=True, color="k")
    >>> res = fit_hist(lambda x,a,b: a+b*x, h) # or fit_hist("a+b*x", h)
    >>> print(res["parnames"],res["parvalues"],res["parerrors"])
    """
    from . import Hist1D

    if draw and not ax:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    xdata_raw = hist.bin_centers
    ydata_raw = hist.counts
    yerrs_raw = hist.errors

    # interlace bin edges with bin centers for smoother fit evaluations
    # `xdata_fine[1::2]` recovers `xdata_raw`
    xdata_fine = np.vstack(
        [hist.edges, np.concatenate([hist.bin_centers, [-1]]),]
    ).T.flatten()[:-1]

    tomask = ((ydata_raw == 0.0) & (yerrs_raw == 0.0) & (not likelihood)) | (
        ~np.isfinite(ydata_raw)
    )
    xdata = xdata_raw[~tomask]
    ydata = ydata_raw[~tomask]
    yerrs = yerrs_raw[~tomask]

    if func == "gaus":
        # gaussian with reasonable initial guesses for parameters
        def func(x, constant=hist.counts.max(), mean=hist.mean(), sigma=hist.std()):
            return constant * np.exp(-((x - mean) ** 2.0) / (2 * sigma ** 2))

    if type(func) in [str]:
        func = expr_to_lambda(func)

    popt, pcov = curve_fit_wrapper(
        func,
        xdata,
        ydata,
        sigma=yerrs,
        absolute_sigma=True,
        likelihood=likelihood,
        **curve_fit_kwargs,
    )

    fit_ydata_fine = func(xdata_fine, *popt)

    if band_style is not None:
        if np.isfinite(pcov).all():
            vopts = np.random.multivariate_normal(popt, pcov, nsamples)
            sampled_ydata_fine = np.vstack(
                [func(xdata_fine, *vopt).T for vopt in vopts]
            )
            sampled_stds_fine = np.nanstd(sampled_ydata_fine, axis=0)
        else:
            import warnings

            warnings.warn("Covariance matrix contains nan/inf")
            sampled_stds_fine = np.ones(len(xdata_fine)) * np.nan
    else:
        sampled_stds_fine = 0.0 * fit_ydata_fine

    hfit = Hist1D.from_bincounts(
        fit_ydata_fine[1::2], hist.edges, errors=sampled_stds_fine[1::2]
    )

    if not likelihood:
        chi2 = ((func(xdata, *popt) - ydata) ** 2.0 / yerrs ** 2.0).sum()
    else:
        chi2 = 0.0
    ndof = len(xdata) - len(popt)

    class wrapper(dict):
        def _repr_html_(self):
            s = "<table><tr><th>parameter</th><th>value</th></tr>"
            for name, x in self["params"].items():
                s += f"<tr><td>{name}</td><td>{x['value']:.4g} &plusmn; {x['error']:.4g}</td></tr>"
            s += "</table>"
            return s

    parnames = func.__code__.co_varnames[1:]
    parvalues = popt
    parerrors = np.diag(pcov) ** 0.5
    params = dict()
    for name, v, e in zip(parnames, parvalues, parerrors):
        params[name] = dict(value=v, error=e)

    res = wrapper(
        params=params,
        parnames=parnames,
        parvalues=parvalues,
        parerrors=parerrors,
        chi2=chi2,
        ndof=ndof,
        hfit=hfit,
    )

    if draw:
        if label:
            label += rf" ($\chi^2$/ndof = {chi2:.3g}/{ndof})"
            for name, x in params.items():
                label += "\n    "
                label += rf"{name} = {x['value']:.3g} $\pm$ {x['error']:.3g}"
        ax.plot(xdata_fine, fit_ydata_fine, color=color, zorder=3, label=label)
        if band_style == "filled":
            ax.fill_between(
                xdata_fine,
                fit_ydata_fine - sampled_stds_fine,
                fit_ydata_fine + sampled_stds_fine,
                facecolor=color,
                alpha=0.25,
                zorder=3,
            )
        elif band_style in ["dashed", "dashdot", "dotted", "solid"]:
            for mult in [-1, 1]:
                ys = fit_ydata_fine + mult * sampled_stds_fine
                ax.plot(xdata_fine, ys, color=color, zorder=3, linestyle=band_style)
        if legend:
            ax.legend()

    return res


def draw_gradient(ax, patches, reverse=False):
    """
    Draws gradient under a step patch (from `histtype="step"`)
    onto specified `ax`.

    Parameters
    ----------
    ax : matplotlib AxesSubplot object
    patches : matplotlib Patch objects
    reverse : bool, default False
        flip the gradient
    """
    import matplotlib.colors as mcolors

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    patch = patches[0]

    color = patch.get_edgecolor()
    zorder = patch.get_zorder()
    alpha = patch.get_alpha() or 1.0

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0.20 * alpha, alpha, 100)[:, None]
    if reverse:
        z[:, :, -1] = z[:, :, -1][::-1]
    im = ax.imshow(
        z, aspect="auto", extent=[xmin, xmax, ymin, ymax], origin="lower", zorder=zorder
    )
    im.set_clip_path(patch)


def plot_stack(hists, **kwargs):
    """
    Plots a list of `Hist1D` objects as a stack

    Parameters
    ----------
    hists : list of `Hist1D` objects
    kwargs : passed to `Hist1D.plot()`
    """
    bottom = 0.0
    for h in hists:
        h.plot(bottom=bottom, **kwargs)
        bottom += h.counts


def darken_color(color, amount=0.2):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], (1.0 - amount) * c[1], c[2])


def draw_error_band(h, ax=None, **kwargs):
    opts = dict(zorder=3, alpha=0.25, step="post")
    if h.metadata.get("color"):
        opts["facecolor"] = h.metadata["color"]
    if kwargs.get("color"):
        opts["facecolor"] = kwargs["color"]
    if "facecolor" in opts:
        opts["facecolor"] = darken_color(opts["facecolor"], 0.2)
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    if h.errors_up is not None:
        ylow = np.concatenate([h.counts - h.errors_down, h.counts[-1:]])
        yhigh = np.concatenate([h.counts + h.errors_up, h.counts[-1:]])
    else:
        ylow = np.concatenate([h.counts - h.errors, h.counts[-1:]])
        yhigh = np.concatenate([h.counts + h.errors, h.counts[-1:]])
    ax.fill_between(h.edges, ylow, yhigh, **opts)


def register_with_dask(classes):
    """
    Register classes with dask so that it can serialize the underlying
    numpy arrays a bit faster
    """
    try:
        from distributed.protocol import register_generic

        for c in classes:
            register_generic(c)
    except:
        pass
