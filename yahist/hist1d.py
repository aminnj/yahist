from __future__ import print_function

import sys
import numpy as np
import copy
import json

from .utils import (
    is_listlike,
    is_datelike,
    convert_dates,
    clopper_pearson_error,
    poisson_errors,
    ignore_division_errors,
    histogramdd_wrapper,
)

from .fit import fit_hist


class Hist1D(object):
    """
    Constructs a Hist1D object from a variety of inputs

    Parameters
    ----------
    obj : a list/array of numbers to histogram, or another `Hist1D` object

    kwargs
        bins : list/array of bin edges, number of bins, string, or "auto", default "auto"
            Follows usage for `np.histogramd`,
            with addition of string specification
        range : list/array of axis ranges, default None
            Follows usage for `np.histogram`
        weights : list/array of weights, default None
            Follows usage for `np.histogram`
        threads : int, default 1
            Number of threads to use for histogramming.
        overflow : bool, default True
            Include overflow counts in outermost bins
        metadata : dict, default {}
            Attach arbitrary extra data to this object

    Returns
    -------
    Hist1D

    Examples
    --------
    >>> x = np.random.normal(0, 1, 1000)
    >>> Hist1D(x, bins=np.linspace(-5,5,11))
    >>> Hist1D(x, bins="10,-5,5")
    >>> Hist1D(x, bins="10,-5,5,20,-3,3")
    >>> h1 = Hist1D(label="foo", color="C0")
    >>> h1 = Hist1D(h1, label="bar", color="C1")
    >>> Hist1D([], metadata=dict(foo=1))
    """

    def __init__(self, obj=[], **kwargs):
        self._counts, self._edges, self._errors = None, None, None
        self._errors_up, self._errors_down = (
            None,
            None,
        )  # used when dividing with binomial errors
        self._metadata = {}
        kwargs = self._extract_metadata(**kwargs)

        if "ROOT." in str(type(obj)):
            self._init_root(obj, **kwargs)
        elif "awkward" in str(type(obj)):
            obj = obj.__array__()
            self._init_numpy(obj, **kwargs)
        elif is_listlike(obj):
            self._init_numpy(obj, **kwargs)
        elif type(obj) is self.__class__:
            # allows Hist1D constructed with another Hist1D to introduce new metadata
            newmetadata = self._metadata.copy()
            self.__dict__.update(obj.__dict__)
            self._metadata.update(newmetadata)
        else:
            raise Exception("empty constructor?")

    def copy(self):
        hnew = self.__class__()
        hnew.__dict__.update(copy.deepcopy(self.__dict__))
        return hnew

    def _init_numpy(
        self, obj, bins="auto", range=None, weights=None, threads=1, overflow=True
    ):

        # convert ROOT-like "50,0,10" to equivalent of np.linspace(0,10,51)
        if isinstance(bins, str) and (bins.count(",") == 2):
            nbins, low, high = bins.split(",")
            range = (float(low), float(high))
            bins = int(nbins)

        if is_datelike(obj):
            obj = convert_dates(obj)
            self._metadata["date_axes"] = ["x"]

        if isinstance(bins, str):

            # if binning integers, binning choice is easy
            if hasattr(obj, "dtype") and ("int" in str(obj.dtype)):
                # check just 10% on each side to get reasonable ranges
                # n = max(int(0.1*len(obj)), 1)
                # maxi = max(obj[:n].max(), obj[-n:].max())
                # mini = min(obj[:n].min(), obj[-n:].min())
                mini, maxi = obj.min(), obj.max()
                bins = np.linspace(mini - 0.5, maxi + 0.5, maxi - mini + 2)
            else:
                bins = np.histogram_bin_edges(obj, bins, range)

        if weights is not None:
            weights = np.array(weights, copy=False)

        if is_datelike(bins):
            bins = convert_dates(bins)
            self._metadata["date_axes"] = ["x"]

        counts, (edges,) = histogramdd_wrapper(
            (obj,), (bins,), (range,), weights, overflow, threads,
        )

        if weights is not None:
            sumw2, _ = histogramdd_wrapper(
                (obj,), (bins,), (range,), weights ** 2, overflow, threads,
            )
            errors = sumw2 ** 0.5
        else:
            errors = counts ** 0.5

        self._counts = counts
        self._edges = edges
        self._errors = errors

    def _init_root(self, obj, **kwargs):
        nbins = obj.GetNbinsX()
        if not kwargs.pop("no_overflow", False):
            # move under and overflow into first and last visible bins
            # set bin error before content because setting the content updates the error?
            obj.SetBinError(
                1, (obj.GetBinError(1) ** 2.0 + obj.GetBinError(0) ** 2.0) ** 0.5
            )
            obj.SetBinError(
                nbins,
                (obj.GetBinError(nbins) ** 2.0 + obj.GetBinError(nbins + 1) ** 2.0)
                ** 0.5,
            )
            obj.SetBinContent(1, obj.GetBinContent(1) + obj.GetBinContent(0))
            obj.SetBinContent(
                nbins, obj.GetBinContent(nbins) + obj.GetBinContent(nbins + 1)
            )
        edges = np.array(
            [1.0 * obj.GetBinLowEdge(ibin) for ibin in range(1, nbins + 2)]
        )
        self._counts = np.array(
            [1.0 * obj.GetBinContent(ibin) for ibin in range(1, nbins + 1)],
            dtype=np.float64,
        )
        self._errors = np.array(
            [1.0 * obj.GetBinError(ibin) for ibin in range(1, nbins + 1)],
            dtype=np.float64,
        )
        self._edges = edges

    def _extract_metadata(self, **kwargs):
        # color and label are special and for convenience, can be specified as top level kwargs
        # e.g., Hist1D(..., color="C0", label="blah", metadata={"foo": "bar"})
        for k in ["color", "label"]:
            if k in kwargs:
                self._metadata[k] = kwargs.pop(k)
        self._metadata.update(kwargs.pop("metadata", dict()))
        return kwargs

    @property
    def metadata(self):
        return self._metadata

    @property
    def errors(self):
        return self._errors

    @property
    def errors_up(self):
        return self._errors_up

    @property
    def errors_down(self):
        return self._errors_down

    @property
    def counts(self):
        return self._counts

    @property
    def edges(self):
        return self._edges

    @property
    def bin_centers(self):
        """
        Returns the midpoints of bin edges.

        Returns
        -------
        array
            Bin centers
        """
        return 0.5 * (self._edges[1:] + self._edges[:-1])

    @property
    def bin_widths(self):
        """
        Returns the widths of bins.

        Returns
        -------
        array
            Bin widths
        """
        return self._edges[1:] - self._edges[:-1]

    @property
    def nbins(self):
        """
        Returns the number of bins

        Returns
        -------
        int
            Number of bins
        """
        return len(self._edges) - 1

    @property
    def dim(self):
        """
        Returns the number of dimensions.
        Hist1D returns 1, Hist2D returns 2

        Returns
        -------
        int
            Number of dimensions
        """
        return self._counts.ndim

    @property
    def integral(self):
        """
        Returns the integral of the histogram (sum of counts).

        Returns
        -------
        float
            Sum of counts
        """
        return self.counts.sum()

    @property
    def integral_error(self):
        """
        Returns the error of the integral of the histogram

        Returns
        -------
        float
            Error on integral
        """
        return (self._errors ** 2.0).sum() ** 0.5

    @property
    def nbytes(self):
        """
        Returns sum of nbytes of underlying numpy arrays

        Returns
        -------
        int
            Number of bytes of underlying numpy arrays
        """
        n = self._counts.nbytes + self._errors.nbytes
        if isinstance(self._edges, tuple):
            for e in self._edges:
                n += e.nbytes
        else:
            n += self._edges.nbytes
        if self._errors_up is not None:
            n += self._errors_up
        if self._errors_down is not None:
            n += self._errors_down
        return n

    def __sizeof__(self):
        return self.nbytes

    def mean(self):
        """
        Returns the mean of the histogram

        Returns
        -------
        float
            Mean of histogram
        """
        return (self.counts * self.bin_centers).sum() / self.integral

    def std(self):
        """
        Returns the standard deviation of the histogram

        Returns
        -------
        float
             standard deviation of histogram (or, RMS)
        """
        variance = (
            self.counts * (self.bin_centers - self.mean()) ** 2.0
        ).sum() / self.integral
        return variance ** 0.5

    def median(self):
        """
        Returns the bin center closest to the median of the histogram.

        Returns
        -------
        float
             median
        """
        return self.quantile(0.5)

    def mode(self):
        """
        Returns mode (bin center for bin with largest value).
        If multiple bins are tied, only the first/leftmost is returned.

        Returns
        -------
        float
             mode
        """
        return self.bin_centers[self.counts.argmax()]

    def _fix_nan(self):
        for x in [self._counts, self._errors, self._errors_up, self._errors_down]:
            if x is not None:
                np.nan_to_num(x, copy=False)

    def _check_consistency(self, other, raise_exception=True):
        if not np.allclose(self._edges, other._edges):
            if raise_exception:
                raise Exception(
                    "These histograms cannot be combined due to different binning"
                )
            else:
                return False
        return True

    def __eq__(self, other):
        if not self._check_consistency(other, raise_exception=False):
            return False
        same = (
            np.allclose(self._counts, other.counts)
            and np.allclose(self._edges, other.edges)
            and np.allclose(self._errors, other.errors)
        )
        if self._errors_up is not None:
            same = same and np.allclose(self._errors_up, other.errors_up)
        if self._errors_down is not None:
            same = same and np.allclose(self._errors_down, other.errors_down)
        return same

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        # allows sum([h1,h2,...]) since start value is 0
        if isinstance(other, int) and (other == 0):
            return self
        if self._counts is None:
            return other
        self._check_consistency(other)
        hnew = self.__class__()
        hnew._counts = self._counts + other._counts
        hnew._errors = (self._errors ** 2.0 + other._errors ** 2.0) ** 0.5
        hnew._edges = self._edges
        hnew._metadata = self._metadata.copy()
        return hnew

    __radd__ = __add__

    def __sub__(self, other):
        self._check_consistency(other)
        hnew = self.__class__()
        hnew._counts = self._counts - other._counts
        hnew._errors = (self._errors ** 2.0 + other._errors ** 2.0) ** 0.5
        hnew._edges = self._edges
        hnew._metadata = self._metadata.copy()
        return hnew

    @ignore_division_errors
    def divide(self, other, binomial=False):
        """
        Divides a histogram object by a scalar or another histogram object (bin-by-bin).

        Parameters
        ----------
        other : float or Hist
        binomial : bool, default False
            Whether to use Clopper-Pearson confidence intervals for errors,
            in which case, the object's `errors_up` and `errors_down` properties
            are filled with asymmetric errors and the `errors` property is
            filled with the average of the two.

        Returns
        -------
        Hist
        """
        self._check_consistency(other)
        hnew = self.__class__()
        hnew._edges = self._edges
        hnew._metadata = self._metadata.copy()
        if not binomial:
            hnew._counts = self._counts / other._counts
            hnew._errors = (
                (self._errors / other._counts) ** 2.0
                + (other._errors * self._counts / (other._counts) ** 2.0) ** 2.0
            ) ** 0.5
            if self._errors_up is not None:
                hnew._errors_up = (
                    (self._errors_up / other._counts) ** 2.0
                    + (other._errors * self._counts / (other._counts) ** 2.0) ** 2.0
                ) ** 0.5
                hnew._errors_down = (
                    (self._errors_down / other._counts) ** 2.0
                    + (other._errors * self._counts / (other._counts) ** 2.0) ** 2.0
                ) ** 0.5
        else:
            bothzero = (self._counts == 0) & (other._counts == 0)
            hnew._errors_down, hnew._errors_up = clopper_pearson_error(
                self._counts, other._counts
            )
            hnew._counts = self._counts / other._counts
            # these are actually the positions for down and up, but we want the errors
            # wrt to the central value
            hnew._errors_up = np.nan_to_num(hnew._errors_up - hnew._counts)
            hnew._errors_down = np.nan_to_num(hnew._counts - hnew._errors_down)

            hnew._errors = 0.5 * (
                hnew._errors_down + hnew._errors_up
            )  # nominal errors are avg of up and down
            # For consistency with TEfficiency, up error is 1 if we have 0/0
            hnew._errors_up[bothzero] = 1.0
        # hnew._fix_nan()
        return hnew

    def __div__(self, other):
        if type(other) in [float, int, np.float64, np.int64]:
            return self.__mul__(1.0 / other)
        elif is_listlike(other):
            # Divide histogram by array (counts) assuming errors are 0
            other = np.array(other)
            if len(other) != len(self._counts):
                raise Exception("Cannot divide due to different binning")
            hnew = self.__class__()
            hnew._edges = self._edges
            hnew._counts = other
            hnew._errors = 0.0 * hnew._counts
            return self.divide(hnew)
        else:
            return self.divide(other)

    __truediv__ = __div__

    def __mul__(self, fact):
        if type(fact) in [float, int, np.float64, np.int64]:
            hnew = self.copy()
            hnew._counts *= fact
            hnew._errors *= fact
            if hnew._errors_up is not None:
                hnew._errors_up *= fact
                hnew._errors_down *= fact
            return hnew
        else:
            raise Exception("Can't multiply histogram by non-scalar")

    __rmul__ = __mul__

    def __pow__(self, expo):
        if type(expo) in [float, int, np.float64, np.int64]:
            hnew = self.copy()
            hnew._counts = hnew._counts ** expo
            hnew._errors *= hnew._counts ** (expo - 1) * expo
            return hnew
        else:
            raise Exception("Can't exponentiate histogram by non-scalar")

    def __repr__(self):
        sep = "\u00B1"
        if sys.version_info[0] < 3:
            sep = sep.encode("utf-8")
        # trick: want to use numpy's smart formatting (truncating,...) of arrays
        # so we convert value,error into a complex number and format that 1D array :)
        prec = np.get_printoptions()["precision"]
        if prec == 8:
            prec = 2
        formatter = {
            "complex_kind": lambda x: "%5.{}f {} %4.{}f".format(prec, sep, prec)
            % (np.real(x), np.imag(x))
        }
        a2s = np.array2string(
            self._counts + self._errors * 1j,
            formatter=formatter,
            suppress_small=True,
            separator="   ",
        )
        return a2s

    def normalize(self, density=False):
        """
        Divides counts of each bin by the sum of the total counts.
        If `density=True`, also divide by bin widths.


        Returns
        -------
        Hist
        """
        if density:
            return self / (self.integral * self.bin_widths)
        else:
            return self / self.integral

    def scale(self, factor):
        """
        Alias for multiplication

        Returns
        -------
        Hist
        """
        return self.__mul__(factor)

    def rebin(self, nrebin):
        """
        Combines adjacent bins by summing contents. The total number
        of bins for each axis must be exactly divisible by `nrebin`.

        Parameters
        ----------
        nrebin : int
            Number of adjacent bins to combine into one bin.

        Returns
        -------
        Hist1D
        """
        nx = self.counts.shape[0]
        bx = nrebin

        if nx % bx != 0:
            raise Exception(
                "This histogram cannot be rebinned since {} is not divisible by {}".format(
                    nx, bx
                )
            )

        counts = self.counts
        edgesx = self.edges
        errors = self.errors

        new_counts = counts.reshape(nx // bx, bx).sum(axis=1)
        new_errors = (errors ** 2).reshape(nx // bx, bx).sum(axis=1) ** 0.5
        new_edgesx = np.append(edgesx[:-1].reshape(nx // bx, -1).T[0], edgesx[-1])

        hnew = self.__class__()
        hnew._edges = new_edgesx
        hnew._errors = new_errors
        hnew._counts = new_counts
        hnew._metadata = self._metadata.copy()
        return hnew

    def restrict(self, low=None, high=None, overflow=False):
        """
        Restricts to a contiguous subset of bins with
        bin center values within [low, high]. If `low`/`high`
        is `None`, there is no lower/upper bound
        Parameters
        ----------
        low : float (default None)
            Lower x center to keep
        high : float (default None)
            Highest x center to keep
        overflow : bool (default False)
            If `True`, adds the excluded bin contents
            into the remaining edge bins.
        Returns
        -------
        Hist1D
        """
        centers = self.bin_centers
        sel = np.ones(self.nbins) > 0.5
        count_low, count_high = 0.0, 0.0
        error2_low, error2_high = 0.0, 0.0
        if low is not None:
            sel &= centers >= low
            count_low = self.counts[centers < low].sum()
            error2_low = (self.errors[centers < low] ** 2).sum()
        if high is not None:
            sel &= centers <= high
            count_high = self.counts[centers > high].sum()
            error2_high = (self.errors[centers > high] ** 2).sum()
        h = self.copy()
        selextra = np.concatenate([sel, [False]])
        selextra[np.argwhere(selextra)[-1][0] + 1] = True
        h._edges = h._edges[selextra]
        h._counts = h._counts[sel]
        h._errors = h._errors[sel]
        if h._errors_up is not None:
            h._errors_up = h._errors_up[sel]
        if h._errors_down is not None:
            h._errors_down = h._errors_down[sel]
        if overflow:
            h._counts[0] += count_low
            h._counts[-1] += count_high
            h._errors[0] = (h._errors[0] ** 2.0 + error2_low) ** 0.5
            h._errors[-1] = (h._errors[-1] ** 2.0 + error2_high) ** 0.5
        return h

    def to_poisson_errors(self, alpha=1 - 0.6827):
        """
        Converts Hist object into one with asymmetric Poissonian errors, inside
        the `errors_up` and `errors_down` properties.

        Parameters
        ----------
        alpha : float, default 1-0.6827
            Confidence interval for errors. 1-sigma by default.

        Returns
        -------
        Hist1D
        """
        lows, highs = poisson_errors(self._counts, alpha=alpha)
        hnew = self.__class__()
        hnew._counts = np.array(self._counts)
        hnew._edges = np.array(self._edges)
        hnew._errors = np.array(self._errors)
        hnew._errors_up = np.array(highs - self._counts)
        hnew._errors_down = np.array(self._counts - lows)
        hnew._metadata = self._metadata.copy()
        return hnew

    def cumulative(self, forward=True):
        """
        Turns Hist object into one with cumulative counts.

        Parameters
        ----------
        forward : bool, default True
            If true, sum the x-axis from low to high, otherwise high to low

        Returns
        -------
        Hist1D
        """
        hnew = self.__class__()
        direction = 1 if forward else -1
        hnew._counts = (self._counts[::direction]).cumsum()[::direction]
        hnew._errors = (self._errors[::direction] ** 2.0).cumsum()[::direction] ** 0.5
        hnew._edges = np.array(self._edges)
        hnew._metadata = self._metadata.copy()
        return hnew

    def lookup(self, x):
        """
        Convert a specified list of x-values into corresponding
        bin counts via `np.digitize`

        Parameters
        ----------
        x : array of x-values, or single x-value

        Returns
        -------
        array
        """
        low = self.edges[0] + self.bin_widths[0] * 0.5
        low = 0.5 * (self.edges[0] + self.edges[1])
        high = 0.5 * (self.edges[-1] + self.edges[-2])
        x = np.clip(x, low, high)
        ibins = np.digitize(x, bins=self.edges) - 1
        return self.counts[ibins]

    def quantile(self, q):
        """
        Returns the bin center corresponding to the quantile(s) `q`.
        Similar to `np.quantile`.

        Parameters
        ----------
        q : float, or array of floats
            quantile between 0 and 1

        Returns
        -------
        float, or array of floats
        """
        counts = np.cumsum(self.counts / self.integral)
        ixs = np.searchsorted(counts, q, side="right")
        return self.bin_centers[ixs]

    def sample(self, size=1e5):
        """
        Returns an array of random samples according
        to a discrete pdf from this histogram.

        Parameters
        ----------
        size : int/float, 1e5
            Number of random values to sample

        Returns
        -------
        array
        """
        cdf = self.normalize().cumulative().counts
        ibins = np.searchsorted(cdf, np.random.rand(int(size)))
        return self.bin_centers[ibins]

    def fill(self, obj, weights=None):
        """
        Fills a `Hist1D`/`Hist2D` in place.

        Parameters
        ----------
        obj : 
            Object to fill, with same definition
            as class construction
        weights : list/array of weights, default None
            See class constructor

        Example
        ----------
        >>> h = Hist1D(bins="10,0,10", label="test")
        >>> h.fill([1,2,3,4])
        >>> h.fill([0,1,2])
        >>> h.median()
        2.5
        """
        h = self + type(self)(obj, bins=self.edges, weights=weights)
        self._counts = h._counts
        self._edges = h._edges
        self._errors = h._errors
        self._errors_up = h._errors_up
        self._errors_down = h._errors_down

    def svg_fast(
        self,
        height=250,
        aspectratio=1.4,
        padding=0.02,
        strokewidth=1,
        color=None,
        bottom=True,
        frame=True,
    ):
        """
        Return HTML svg tag with bare-bones version of histogram
        (no ticks, labels).

        Parameters
        ----------
        height : int, default 250
            Height of plot in pixels
        padding : float, default 0.025
            Fraction of height or width to keep between edges of plot and svg view size
        aspectratio : float, default 1.4
            Aspect ratio of plot
        strokewidth : float, default 1
            Width of strokes
        bottom : bool, default True
            Draw line at the bottom
        color : str, default None",
            Stroke color and fill color (with 15% opacity)
            If color is in the histogram metadata, it will take precedence. 
        frame : bool, default True
            Draw frame/border

        Returns
        -------
        str
        """
        import matplotlib.colors
        import uuid

        if color is None:
            if self.metadata.get("color") is not None:
                color = self.metadata["color"]
            else:
                color = "C0"
        color = matplotlib.colors.to_hex(color)

        uid = str(uuid.uuid4()).split("-")[0]

        width = height * aspectratio

        safecounts = np.array(self._counts)
        safecounts[~np.isfinite(safecounts)] = 0.0

        # map 0,max -> height-padding*height,0+padding*height
        ys = height * (
            (2 * padding - 1) / safecounts.max() * safecounts + (1 - padding)
        )
        # map min,max -> padding*width,width-padding*width
        xs = width * (
            (1 - 2 * padding)
            / (self._edges.max() - self._edges.min())
            * (self._edges - self._edges.min())
            + padding
        )

        points = []
        points.append([padding * width, height * (1 - padding)])
        for i in range(len(xs) - 1):
            points.append([xs[i], ys[i]])
            points.append([xs[i + 1], ys[i]])
        points.append([width * (1 - padding), height * (1 - padding)])
        if bottom:
            points.append([padding * width, height * (1 - padding)])

        pathstr = " ".join("{},{}".format(*p) for p in points)

        if frame:
            framestr = """<rect width="{width}" height="{height}" fill="none" stroke="#000" stroke-width="2" />""".format(
                width=width, height=height
            )
        else:
            framestr = ""
        source = """
        <svg width="{width}" height="{height}" version="1.1" xmlns="http://www.w3.org/2000/svg">
          <defs>
          <linearGradient id="grad_{uid}" x2="0" y2="1">
            <stop offset="0%" stop-color="{color}"/>
            <stop offset="30%" stop-color="{color}"/>
            <stop offset="100%" stop-color="#ffffff00"/>
          </linearGradient>
          </defs>
          {framestr}
          <polyline points="{pathstr}" stroke="{color}" fill="{fill}" fill-opacity="0.15" stroke-width="{strokewidth}"/>
        </svg>
        """.format(
            uid=uid,
            width=width,
            framestr=framestr,
            height=height,
            pathstr=pathstr,
            strokewidth=strokewidth,
            color=color,
            fill=color if bottom else "url(#grad_{uid})".format(uid=uid),
        )
        return source

    def svg(self, **kwargs):
        """
        Return HTML svg tag with Matplotlib-rendered svg.

        Parameters
        ----------
        **kwargs
            Parameters to be passed to `self.plot()` function.

        Returns
        -------
        str
        """
        from io import BytesIO
        import matplotlib.pyplot as plt
        import base64

        fig, ax = plt.subplots(figsize=(4, 3))
        fig.subplots_adjust(bottom=0.15, right=0.95, top=0.94)
        self.plot(ax=ax, histtype="step", **kwargs)
        buf = BytesIO()
        fig.savefig(buf, format="svg")
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        src = "<img src='data:image/svg+xml;base64,{}'/>".format(data)
        return src

    def html_table(self, suppress=True):
        """
        Return HTML table tag with bin contents (counts and errors)
        compactly formatted. Only the four leftmost and rightmost
        bins are shown, while the rest are hidden.

        Parameters
        ----------
        suppress : bool, default True
            if True, hide middle bins/rows

        Returns
        -------
        str
        """
        tablerows = []
        nrows = len(self._counts)
        ntoshow = (
            4 if suppress else self.nbins // 2
        )  # num of start and end rows to show

        def format_row(low, high, count, error):
            return "<tr><td>({:g},{:g})</td><td>{:g} \u00B1 {:g}</td></tr>".format(
                low, high, count, error
            )

        # NOTE, can be optimized: we don't need to convert every row if we will hide some later
        for lhce in zip(self._edges[:-1], self._edges[1:], self._counts, self._errors):
            tablerows.append(format_row(*lhce))
        if nrows < ntoshow * 2 + 2:  # we don't ever want to hide just 1 row
            tablestr = " ".join(tablerows)
        else:
            nhidden = nrows - ntoshow * 2  # number hidden in the middle
            tablerows = (
                tablerows[:ntoshow]
                + [
                    "<tr><td colspan='2'><center>[{} rows hidden]</center></td></tr>".format(
                        nhidden
                    )
                ]
                + tablerows[-ntoshow:]
            )
            tablestr = "\n".join(tablerows)
        return """
                <table style='border:1px solid black;'">
                    <thead><tr><th>bin</th><th>content</th></tr></thead>
                    {tablestr}
                </table>
            """.format(
            tablestr=tablestr
        )

    def _repr_html_(self):
        tablestr = self.html_table()
        svgsource = self.svg()

        source = """
        <div style="max-height:1000px;max-width:1500px;overflow:auto">
        <b>total count</b>: {count}, <b>metadata</b>: {metadata}<br>
        <div style="display:flex;">
            <div style="display:inline;">
                {tablestr}
            </div>
            <div style="display:inline; margin: auto 2%;">
                {svgsource}
            </div>
            </div>
        </div>
        """.format(
            count=self._counts.sum(),
            metadata=self._metadata,
            svgsource=svgsource,
            tablestr=tablestr,
        )
        return source

    def to_json(self, obj=None):
        """
        Returns json-serialized version of this object.

        Parameters
        ----------
        obj : str, default None
            If specified, writes json to path instead of returning string.
            If the path ends with '.gz', compresses with gzip.

        Returns
        -------
        str
        """

        def default(obj):
            if hasattr(obj, "__array__"):
                return obj.tolist()
            raise TypeError("Don't know how to serialize object of type", type(obj))

        s = json.dumps(self.__dict__, default=default)
        if obj is None:
            return s
        else:
            opener = open
            mode = "w"
            if obj.endswith(".gz"):
                import gzip

                opener = gzip.open
                mode = "wb"
                s = s.encode()
            with opener(obj, mode) as fh:
                fh.write(s)

    @classmethod
    def from_json(cls, obj):
        """
        Converts serialized json to histogram object.

        Parameters
        ----------
        obj : str
            json-serialized object from `self.to_json()` or file path

        Returns
        -------
        Hist
        """
        if obj.startswith("{"):
            obj = json.loads(obj)
        else:
            opener = open
            mode = "r"
            if obj.endswith(".gz"):
                import gzip

                opener = gzip.open
                mode = "rb"
            with opener(obj, mode) as fh:
                obj = json.load(fh)
        for k in obj:
            if is_listlike(obj[k]):
                v = np.array(obj[k])
                if (k in ["_edges"]) and (v.dtype == "O"):
                    v = [np.array(x) for x in obj[k]]
                obj[k] = v
        hnew = cls()
        hnew.__dict__.update(obj)
        return hnew

    @classmethod
    def from_bincounts(cls, counts, bins=None, errors=None, **kwargs):
        """
        Creates histogram object from array of histogrammed counts,
        edges/bins, and optionally errors.

        Parameters
        ----------
        counts : array
            Array of bin counts
        bins : array, default None
            Array of bin edges. If not specified for Hist1D,
            uses `bins = np.arange(len(counts)+1)`.
        errors : array, default None
            Array of bin errors (optional)
        **kwargs
            Parameters to be passed to `Hist1D`/`Hist2D` constructor.

        Returns
        -------
        Hist
        """
        hnew = cls(**kwargs)
        counts = np.asarray(counts)
        if cls.__name__ == "Hist1D":
            if bins is None:
                bins = np.arange(len(counts) + 1)
            else:
                bins = np.asarray(bins)

        hnew._counts = counts.astype(np.float64)
        hnew._edges = np.asarray(bins)
        if errors is not None:
            hnew._errors = np.asarray(errors)
        else:
            hnew._errors = hnew._counts ** 0.5
        return hnew

    @classmethod
    def from_random(
        cls, which="norm", params=[0.0, 1.0], size=1e5, random_state=None, **kwargs
    ):
        """
        Creates histogram object from random values of
        a given distribution within `scipy.stats`

        Parameters
        ----------
        which : str, default "norm"
            Distribution within `scipy.stats`
        params : list/array, default [0, 1]
            Parameters to distribution
        size : int/float, 1e5
            Number of random values to sample/fill histogram
        random_state : int, default None

        Returns
        -------
        Hist
        """
        import scipy.stats

        try:
            func = getattr(scipy.stats, which)
        except AttributeError:
            valid = sorted(
                [x for x in dir(scipy.stats) if hasattr(getattr(scipy.stats, x), "rvs")]
            )
            raise Exception(
                f"{which} is not a valid distribution in `scipy.stats`. Valid distributions are: {valid}"
            )
        if isinstance(size, float):
            size = int(size)
        if (
            "multivariate" not in which
            and cls.__name__ == "Hist2D"
            and not is_listlike(size)
        ):
            size = (size, 2)
        v = func(*params).rvs(size=size, random_state=random_state)
        h = cls(v, **kwargs)
        return h

    def plot(
        self,
        ax=None,
        histtype="step",
        legend=True,
        counts=False,
        errors=False,
        fmt="o",
        label=None,
        color=None,
        counts_formatter="{:3g}".format,
        counts_fontsize=10,
        interactive=False,
        **kwargs,
    ):
        """
        Plot this histogram object using matplotlib's `hist`
        function, or `errorbar` (depending on the value of the `errors` argument).

        Parameters
        ----------
        ax : matplotlib AxesSubplot object, default None
            matplotlib AxesSubplot object. Created if `None`.
        color : str, default None
            If None, uses default matplotlib color cycler
        counts, bool False
            If True, show text labels for counts (and/or errors). See
            `counts_formatter` and `counts_fontsize`.
        counts_formatter : callable, default `"{:3g}".format`
            Two-parameter function used to format count and error labels.
            Thus, if a second placeholder is specified (e.g., `"{:3g} +- {:3g}".format`),
            the bin error can be shown as well.
        counts_fontsize
            Font size of count labels
        errors, bool False
            If True, plot markers with error bars (`ax.errorbar()`) instead of `ax.hist()`.
        fmt : str, default "o"
            `fmt` kwarg used for matplotlib plotting
        label : str, default None
            Label for legend entry
        interactive : bool, default False
            Use plotly to make an interactive plot. See `Hist1D.plot_plotly()`.
        legend : bool, default True
            If True and the histogram has a label, draw the legend
        **kwargs
            Parameters to be passed to matplotlib
            or `errorbar` (if `errors=True`) `hist` (otherwise) function.


        Returns
        -------
        matplotlib AxesSubplot object
        """

        if interactive:
            return self.plot_plotly(errors=errors, color=color, label=label, **kwargs,)

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if (color is None) and (self.metadata.get("color") is not None):
            color = self.metadata["color"]

        if (label is None) and (self.metadata.get("label") is not None):
            label = self.metadata["label"]

        show_counts = counts or kwargs.pop("show_counts", False)
        show_errors = errors or kwargs.pop("show_errors", False)

        counts = self._counts
        edges = self._edges
        yerrs = self._errors
        xerrs = 0.5 * self.bin_widths
        mask = ((counts != 0.0) | (yerrs != 0.0)) & np.isfinite(counts)
        centers = self.bin_centers

        if show_errors:
            yerr = yerrs[mask]
            if self.errors_up is not None:
                yerr = self.errors_down[mask], self.errors_up[mask]
            patches = ax.errorbar(
                centers[mask],
                counts[mask],
                xerr=xerrs[mask],
                yerr=yerr,
                fmt=fmt,
                color=color,
                label=label,
                **kwargs,
            )
            # If there are points with values of 0, they are not drawn
            # and the xlims will be compressed, so we force the bounds
            ax.set_xlim(self._edges[0], self._edges[-1])
        else:
            _, _, patches = ax.hist(
                centers[mask],
                edges,
                weights=counts[mask],
                histtype=histtype,
                color=color,
                label=label,
                **kwargs,
            )

        if label and legend:
            ax.legend()

        if show_counts:
            patch = patches[0]
            color = None
            if color is None and hasattr(patch, "get_facecolor"):
                color = patch.get_facecolor()
                if color[-1] == 0.0:
                    color = None
            if color is None and hasattr(patch, "get_color"):
                color = patch.get_color()
                if color[-1] == 0.0:
                    color = None
            if color is None and hasattr(patch, "get_edgecolor"):
                color = patch.get_edgecolor()
            xtodraw = centers[mask]
            ytexts = counts[mask]
            yerrtexts = yerrs[mask]
            if show_errors:
                ytodraw = counts[mask] + yerrs[mask]
            else:
                ytodraw = counts[mask]
            for xpos, ypos, ytext, yerrtext in zip(xtodraw, ytodraw, ytexts, yerrtexts):
                ax.text(
                    xpos,
                    ypos,
                    counts_formatter(ytext, yerrtext),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=counts_fontsize,
                    color=color,
                )

        if "date_axes" in self.metadata:
            import matplotlib.dates

            locator = matplotlib.dates.AutoDateLocator()
            formatter = matplotlib.dates.ConciseDateFormatter(locator)
            which_axes = self.metadata["date_axes"]
            if "x" in which_axes:
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

        return ax

    def plot_plotly(
        self,
        fig=None,
        color=None,
        errors=False,
        log=False,
        label=None,
        flipxy=False,
        alpha=1,
        stack=False,
        **kwargs,
    ):
        import plotly.graph_objects as go
        import matplotlib.colors

        if color is None:
            if self.metadata.get("color") is not None:
                color = self.metadata["color"]
            else:
                color = "C0"
        color = matplotlib.colors.to_hex(color)
        yscale = "log" if log else "linear"
        rangemode = "nonnegative"
        if np.any(self.counts < 0.0):
            rangemode = "normal"
        if errors:
            trace = go.Scatter()
            trace.error_x = dict(array=self.bin_widths / 2, width=0, color=color)
            trace.error_y = dict(array=self.errors, width=0, color=color)
            if flipxy:
                trace.error_x, trace.error_y = (
                    trace.error_y.to_plotly_json(),
                    trace.error_x.to_plotly_json(),
                )
            trace.mode = "markers"
            trace.marker.size = 5
        else:
            trace = go.Bar()
            trace.width = self.bin_widths
            trace.orientation = "h" if flipxy else "v"
        trace.marker.color = color
        trace.marker.line.width = 0.0
        trace.x = self.bin_centers
        trace.y = self.counts
        trace.opacity = alpha
        if flipxy:
            trace.x, trace.y = trace.y, trace.x
        if label is not None:
            trace.name = label
        elif self.metadata.get("label") is not None:
            trace.name = self.metadata["label"]
        else:
            trace.showlegend = False

        if fig is None:
            fig = go.Figure()
        fig.add_trace(trace)
        fig.update_layout(
            bargap=0,
            barmode="stack" if stack else "overlay",
            height=300,
            width=400,
            template="simple_white",
            font_family="Arial",
            xaxis=dict(mirror=True),
            yaxis=dict(mirror=True, type=yscale, rangemode=rangemode),
            margin=dict(l=10, r=10, b=10, t=30, pad=0,),
        )
        return fig

    def fit(self, func, **kwargs):
        return fit_hist(func, self, **kwargs)

    fit.__doc__ = fit_hist.__doc__
