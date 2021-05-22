import re
import numpy as np
import matplotlib

import matplotlib.patches
import matplotlib.lines
import matplotlib.legend

from .utils import is_listlike


class BandObject(matplotlib.patches.Rectangle):
    pass


class BandObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        color = orig_handle.get_facecolor()
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = matplotlib.patches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=color,
            edgecolor="none",
            lw=0.0,
            alpha=0.25,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        patch = matplotlib.lines.Line2D(
            [x0 + width * 0.03, x0 + width - width * 0.03],
            [y0 + height * 0.5],
            color=color,
            linewidth=1,
            linestyle="-",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


def expr_to_lambda(expr):
    """
    Converts a string expression like
        "a+b*np.exp(-c*x+math.pi)"
    into a lambda function with 1 variable and N parameters,
        lambda x,a,b,c: "a+b*np.exp(-c*x+math.pi)"
    `x` is assumed to be the main variable.
    Very simple logic that ignores things like `foo.bar`
    or `foo(` from being considered a parameter.

    Parameters
    ----------
    expr : str

    Returns
    -------
    callable/lambda
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
    extent=None,
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
    extent: 2-tuple, default None
        if 2-tuple, these are used with `Hist1D.restrict()` to
        fit only a subset of the x-axis (but still draw the full range)
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
        - a callable function "func" corresponding to the input function
          but with fitted parameters

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

    hist_full = hist.copy()

    if is_listlike(extent) and len(extent) == 2:
        hist = hist.restrict(*extent)

    xdata_raw = hist.bin_centers
    ydata_raw = hist.counts
    yerrs_raw = hist.errors

    # interlace bin edges with bin centers for smoother fit evaluations
    # `xdata_fine[1::2]` recovers `xdata_raw`
    xdata_fine = np.vstack(
        [hist_full.edges, np.concatenate([hist_full.bin_centers, [-1]]),]
    ).T.flatten()[:-1]

    tomask = ((ydata_raw == 0.0) & (yerrs_raw == 0.0) & (not likelihood)) | (
        ~np.isfinite(ydata_raw)
    )
    xdata = xdata_raw[~tomask]
    ydata = ydata_raw[~tomask]
    yerrs = yerrs_raw[~tomask]

    if isinstance(func, str):
        func_parts = re.split(r"(\d+|\W+)", func)
        if func == "gaus":
            # gaussian with reasonable initial guesses for parameters
            def func(
                x, constant=hist.counts.max(), mean=hist.median(), sigma=hist.std()
            ):
                return constant * np.exp(-((x - mean) ** 2.0) / (2 * sigma ** 2))

        elif "gaus" in func_parts:
            func_parts[
                func_parts.index("gaus")
            ] = "constant * np.exp(-((x - mean) ** 2.0) / (2 * sigma ** 2))"
            func = expr_to_lambda("".join(func_parts))
        else:
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
        fit_ydata_fine[1::2], hist_full.edges, errors=sampled_stds_fine[1::2]
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
        func=lambda x: func(x, *parvalues),
    )

    if draw:
        if label:
            # label += rf" ($\chi^2$/ndof = {chi2:.3g}/{ndof})"
            for name, x in params.items():
                label += "\n    "
                label += rf"{name} = {x['value']:.3g} $\pm$ {x['error']:.3g}"
        ax.plot(xdata_fine, fit_ydata_fine, color=color, zorder=3)
        if band_style == "filled":
            ax.fill_between(
                xdata_fine,
                fit_ydata_fine - sampled_stds_fine,
                fit_ydata_fine + sampled_stds_fine,
                facecolor=color,
                alpha=0.25,
                zorder=3,
            )
            matplotlib.legend.Legend.update_default_handler_map(
                {BandObject: BandObjectHandler()}
            )
            ax.add_patch(
                BandObject((0, 0), 0, 0, label=label, color=color, visible=False)
            )

        elif band_style in ["dashed", "dashdot", "dotted", "solid"]:
            for mult in [-1, 1]:
                ys = fit_ydata_fine + mult * sampled_stds_fine
                ax.plot(xdata_fine, ys, color=color, zorder=3, linestyle=band_style)
        if legend:
            ax.legend()

    return res
