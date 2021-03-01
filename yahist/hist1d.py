from __future__ import print_function

import sys
import numpy as np
import copy
import json
import base64

from .utils import (
    is_listlike,
    is_datelike,
    convert_dates,
    clopper_pearson_error,
    poisson_errors,
    ignore_division_errors,
    has_uniform_spacing,
    fit_hist,
    draw_gradient,
)


class Hist1D(object):
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
            if len(obj) == 0:
                self._empty = True
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

    def _init_numpy(self, obj, **kwargs):
        kwargs["bins"] = kwargs.get("bins", "auto")

        if ("weights" in kwargs) and (kwargs["weights"] is None):
            kwargs.pop("weights")

        if kwargs.pop("norm", False) or kwargs.pop("density", False):
            raise Exception(
                "Please use the .normalize() [.normalize(density=True)] method on the histogram object."
            )
        if kwargs.pop("cumulative", False):
            raise Exception(
                "Please use the .cumulative() method on the histogram object."
            )

        # convert ROOT-like "50,0,10" to np.linspace(0,10,51)
        if isinstance(kwargs.get("bins"), str) and (kwargs["bins"].count(",") == 2):
            nbins, low, high = kwargs["bins"].split(",")
            kwargs["range"] = (float(low), float(high))
            kwargs["bins"] = np.linspace(float(low), float(high), int(nbins) + 1)

        if is_datelike(obj):
            obj = convert_dates(obj)
            self._metadata["date_axes"] = ["x"]

        if is_listlike(kwargs.get("bins")):
            bins = kwargs["bins"]

            if is_datelike(bins):
                bins = convert_dates(bins)
                self._metadata["date_axes"] = ["x"]

            if kwargs.pop("overflow", True):
                clip_low = 0.5 * (bins[0] + bins[1])
                clip_high = 0.5 * (bins[-2] + bins[-1])
                obj = np.clip(obj, clip_low, clip_high)
            if has_uniform_spacing(bins):
                # if uniformly spaced, use O(N) algorithm instead of O(NlogN) (binary search/`np.searchsorted`)
                # inside numpy by specifying number of bins and range
                kwargs["range"] = (bins[0], bins[-1])
                kwargs["bins"] = len(bins) - 1
        else:
            kwargs.pop("overflow", None)
        self._counts, self._edges = np.histogram(obj, **kwargs)
        self._counts = self._counts.astype(np.float64)

        # poisson defaults if not specified
        if self._errors is None:
            if "weights" not in kwargs:
                self._errors = np.sqrt(self._counts)
            else:
                # if weighted entries, need to get sum of sq. weights per bin
                # and sqrt of that is bin error
                kwargs["weights"] = kwargs["weights"] ** 2.0
                counts, _ = np.histogram(obj, **kwargs)
                self._errors = np.sqrt(counts)
        self._errors = self._errors.astype(np.float64)

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
        if (type(other) is int) and (other == 0):
            return self
        if hasattr(self, "_empty"):
            return other
        if hasattr(other, "_empty"):
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
            hnew._errors_up = hnew._errors_up - hnew._counts
            hnew._errors_down = hnew._counts - hnew._errors_down
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

    def restrict(self, low=None, high=None):
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

        Returns
        -------
        Hist1D
        """
        centers = self.bin_centers
        sel = np.ones(self.nbins) > 0.5
        if low is not None:
            sel &= centers >= low
        if high is not None:
            sel &= centers <= high
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

    def svg_fast(
        self,
        height=250,
        aspectratio=1.4,
        padding=0.02,
        strokewidth=1,
        color="#4285F4",
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
        color : str, default "#4285F4",
            Stroke color and fill color (with 15% opacity)
        frame : bool, default True
            Draw frame/border

        Returns
        -------
        str
        """
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
          {framestr}
          <polyline points="{pathstr}" stroke="{color}" fill="{color}" fill-opacity="0.15" stroke-width="{strokewidth}"/>
        </svg>
        """.format(
            width=width,
            framestr=framestr,
            height=height,
            pathstr=pathstr,
            strokewidth=strokewidth,
            color=color,
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

        fig, ax = plt.subplots(figsize=(4, 3))
        fig.subplots_adjust(bottom=0.08, right=0.99, top=0.99)
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
            if specified, writes json to path instead of returning string

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
            with open(obj, "w") as fh:
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
            with open(obj, "r") as fh:
                obj = json.load(fh)
        for k in obj:
            if is_listlike(obj[k]):
                obj[k] = np.array(obj[k])
        hnew = cls()
        hnew.__dict__.update(obj)
        return hnew

    @classmethod
    def from_bincounts(cls, counts, bins=None, errors=None):
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

        Returns
        -------
        Hist
        """
        hnew = cls()
        counts = np.asarray(counts)
        if cls.__name__ == "Hist1D":
            if bins is None:
                bins = np.arange(len(counts) + 1)
            else:
                bins = np.asarray(bins)

        hnew._counts = counts.astype(np.float64)
        hnew._edges = bins
        if errors is not None:
            hnew._errors = errors
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
        if type(size) == float:
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

    def plot(self, ax=None, **kwargs):
        """
        Plot this histogram object using matplotlib's `hist`
        function, or `errorbar`.

        Parameters
        ----------
        ax : matplotlib AxesSubplot object, default None
            matplotlib AxesSubplot object. Created if `None`.
        counts
            Alias for `show_counts`
        counts_fmt_func : function, default "{:3g}".format
            Function used to format count labels
        counts_fontsize
            Font size of count labels
        errors
            Alias for `show_errors`
        fmt : str, default "o"
            `fmt` kwarg used for matplotlib plotting
        gradient : bool, default False
            fill a light gradient under histogram if `histtype="step"`
        interactive : bool, default False
            Use plotly to make an interactive plot
        show_counts : bool, default False
            Show count labels for each bin
        show_errors : bool, default False
            Show error bars
        legend : bool, default True
            if True and the histogram has a label, draw the legend
        return_self : bool, default False
            If true, return self (Hist1D object)
        **kwargs
            Parameters to be passed to matplotlib
            `hist` or `errorbar` function.


        Returns
        -------
        Hist1D (self) if `return_self` is True, otherwise
        matplotlib AxesSubplot object
        """

        if kwargs.pop("interactive", False):
            return self.plot_plotly(**kwargs)

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        kwargs["color"] = kwargs.get("color", self.metadata.get("color"))
        kwargs["label"] = kwargs.get("label", self.metadata.get("label"))
        kwargs["histtype"] = kwargs.get("histtype", "step")
        legend = kwargs.pop("legend", True)
        show_counts = kwargs.pop("show_counts", kwargs.pop("counts", False))
        show_errors = kwargs.pop("show_errors", kwargs.pop("errors", False))
        counts_fmt_func = kwargs.pop("counts_fmt_func", "{:3g}".format)
        counts_fontsize = kwargs.pop("counts_fontsize", 10)
        gradient = kwargs.pop("gradient", False)
        return_self = kwargs.pop("return_self", False)
        counts = self._counts
        edges = self._edges
        yerrs = self._errors
        xerrs = 0.5 * self.bin_widths
        mask = ((counts != 0.0) | (yerrs != 0.0)) & np.isfinite(counts)
        centers = self.bin_centers

        if gradient:
            kwargs["histtype"] = "step"

        if show_errors:
            kwargs["fmt"] = kwargs.get("fmt", "o")
            kwargs.pop("histtype", None)
            yerr = yerrs[mask]
            if self.errors_up is not None:
                yerr = self.errors_down[mask], self.errors_up[mask]
            patches = ax.errorbar(
                centers[mask], counts[mask], xerr=xerrs[mask], yerr=yerr, **kwargs
            )
            # If there are points with values of 0, they are not drawn
            # and the xlims will be compressed, so we force the bounds
            ax.set_xlim(self._edges[0], self._edges[-1])
        else:
            _, _, patches = ax.hist(
                centers[mask], edges, weights=counts[mask], **kwargs
            )

            if gradient:
                draw_gradient(
                    ax, patches, reverse=(kwargs.get("histtype", "") == "stepfilled")
                )

        if kwargs["label"] and legend:
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
            if show_errors:
                ytodraw = counts[mask] + yerrs[mask]
            else:
                ytodraw = ytexts
            for x, y, ytext in zip(xtodraw, ytodraw, ytexts):
                ax.text(
                    x,
                    y,
                    counts_fmt_func(ytext),
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

        if return_self:
            return self
        else:
            return ax

    def plot_plotly(self, **kwargs):
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Bar(
                x=self.bin_centers,
                width=self.bin_widths,
                y=self.counts,
                marker=dict(line=dict(width=0,)),
            ),
        )
        fig.update_layout(
            bargap=0,
            height=300,
            width=400,
            template="simple_white",
            font_family="Arial",
            xaxis=dict(mirror=True,),
            yaxis=dict(
                mirror=True, type=("log" if kwargs.get("log", False) else "linear"),
            ),
            margin=dict(l=10, r=10, b=10, t=30, pad=0,),
        )
        return fig

    def fit(self, func, **kwargs):
        return fit_hist(func, self, **kwargs)

    fit.__doc__ = fit_hist.__doc__
