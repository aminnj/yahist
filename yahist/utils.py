import re

import matplotlib
import numpy as np

import boost_histogram as bh


def is_listlike(obj):
    return (type(obj) in [tuple, list]) or (np.ndim(obj) >= 1)


def is_datelike(obj):
    if hasattr(obj, "dtype"):
        # return isinstance(obj.dtype.type, (type(np.datetime64),))
        return obj.dtype.type == np.datetime64
    elif "Timestamp" in str(type(obj)):
        return True
    else:
        return False


def convert_dates(obj):
    import matplotlib.dates

    return matplotlib.dates.date2num(obj)


def has_uniform_spacing(obj, epsilon=1e-6):
    offsets = np.ediff1d(obj)
    return np.all(np.abs(offsets - offsets[0]) < epsilon)


def histogramdd_wrapper(a, bins, range_, weights, overflow, threads):
    # Based on
    # https://github.com/scikit-hep/boost-histogram/blob/develop/src/boost_histogram/numpy.py
    if isinstance(a, np.ndarray):
        a = a.T

    rank = len(a)
    try:
        bins = (int(bins),) * rank
    except TypeError:
        pass

    if range_ is None:
        range_ = (None,) * rank

    axs = []
    for n, (b, r) in enumerate(zip(bins, range_)):
        if is_listlike(b) and has_uniform_spacing(b):
            r = b[0], b[-1]
            b = len(b) - 1
        if np.issubdtype(type(b), np.integer):
            if r is None:
                if len(a[n]):
                    r = (np.min(a[n]), np.max(a[n]))
                else:
                    r = (0, 1)
            axis = bh.axis.Regular(b, r[0], r[1], underflow=True, overflow=True)
            # low, high = r[0], r[1]
            # integer_binning = float(low).is_integer() and (float(high-low) == float(b))
            # if integer_binning:
            #     high = np.nextafter(high, np.finfo("d").max)
            #     axis = bh.axis.Integer(int(low), int(high), underflow=True, overflow=True)
            # else:
            #    axis = bh.axis.Regular(b, low, high, underflow=True, overflow=True)
            axs.append(axis)
        else:
            barr = np.asarray(b, dtype=np.double)
            axs.append(bh.axis.Variable(barr, underflow=True, overflow=True))

    counts, edges = (
        bh.Histogram(*axs)
        .fill(*a, weight=weights, threads=threads)
        .to_numpy(view=True, dd=True, flow=overflow)
    )

    if overflow:
        edges = list(edges)
        for n in range(rank):
            edges[n] = edges[n][1:-1]

        if np.ndim(counts) == 1:
            counts[1] += counts[0]
            counts[-2] += counts[-1]
            counts = counts[1:-1]
        else:
            counts[:, 1] += counts[:, 0]
            counts[:, -2] += counts[:, -1]
            counts[1, :] += counts[0, :]
            counts[-2, :] += counts[-1, :]
            counts = counts[1:-1, 1:-1]

    return counts, edges


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
    matching TEfficiency::ClopperPearson(),
    >>> ROOT.TEfficiency.ClopperPearson(total, passed, level, is_upper)
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
