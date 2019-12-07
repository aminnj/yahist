from __future__ import print_function

import sys
import numpy as np
import copy
import json
import base64

from .utils import (
    is_listlike,
    clopper_pearson_error,
    poisson_errors,
    ignore_division_errors,
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
        if is_listlike(obj):
            self._init_numpy(obj, **kwargs)
        elif type(obj) is self.__class__:
            # allows Hist1D constructed with another Hist1D to introduce new metadata
            newmetadata = self._metadata.copy()
            self.__dict__.update(obj.__dict__)
            self._metadata.update(newmetadata)
        else:
            raise Exception("empty constructor?")

    def _copy(self):
        hnew = self.__class__()
        hnew.__dict__.update(copy.deepcopy(self.__dict__))
        return hnew

    def _init_numpy(self, obj, **kwargs):
        if isinstance(kwargs.get("bins"), str):
            if kwargs["bins"].count(",") == 2:
                nbins, low, high = kwargs["bins"].split(",")
                kwargs["bins"] = np.linspace(float(low), float(high), int(nbins) + 1)
        if (
            kwargs.pop("overflow", True)
            and ("bins" in kwargs)
            and not isinstance(kwargs["bins"], str)
        ):
            bins = kwargs["bins"]
            clip_low = 0.5 * (bins[0] + bins[1])
            clip_high = 0.5 * (bins[-2] + bins[-1])
            obj = np.clip(obj, clip_low, clip_high)
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

    def _extract_metadata(self, **kwargs):
        for k in ["color", "label"]:
            if k in kwargs:
                self._metadata[k] = kwargs.pop(k)
        return kwargs

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
        return 0.5 * (self._edges[1:] + self._edges[:-1])

    @property
    def bin_widths(self):
        return self._edges[1:] - self._edges[:-1]

    @property
    def integral(self):
        return self.counts.sum()

    @property
    def integral_error(self):
        return (self._errors ** 2.0).sum() ** 0.5

    def _fix_nan(self):
        for x in [self._counts, self._errors, self._errors_up, self._errors_down]:
            if x is not None:
                np.nan_to_num(x, copy=False)

    def _check_consistency(self, other, raise_exception=True):
        if len(self._edges) != len(other._edges):
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
        return (
            np.allclose(self._counts, other.counts)
            and np.allclose(self._edges, other.edges)
            and np.allclose(self._errors, other.errors)
            and (
                (self._errors_up is not None)
                or np.allclose(self._errors_up, other.errors_up)
            )
            and (
                (self._errors_down is not None)
                or np.allclose(self._errors_down, other.errors_down)
            )
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        if other is 0:
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
            hnew = self._copy()
            hnew._counts *= fact
            hnew._errors *= fact
            return hnew
        else:
            raise Exception("Can't multiply histogram by non-scalar")

    __rmul__ = __mul__

    def __pow__(self, expo):
        if type(expo) in [float, int, np.float64, np.int64]:
            hnew = self._copy()
            hnew._counts = hnew._counts ** expo
            hnew._errors *= hnew._counts ** (expo - 1) * expo
            return hnew
        else:
            raise Exception("Can't exponentiate histogram by non-scalar")

    def __repr__(self):
        sep = u"\u00B1"
        if sys.version_info[0] < 3:
            sep = sep.encode("utf-8")
        # trick: want to use numpy's smart formatting (truncating,...) of arrays
        # so we convert value,error into a complex number and format that 1D array :)
        prec = np.get_printoptions()["precision"]
        if prec == 8:
            prec = 3
        formatter = {
            "complex_kind": lambda x: "%5.{}f {} %4.{}f".format(prec, sep, prec)
            % (np.real(x), np.imag(x))
        }
        # formatter = {"complex_kind": lambda x:("{:g} %s {:g}" % (sep)).format(np.real(x),np.imag(x))}
        a2s = np.array2string(
            self._counts + self._errors * 1j,
            formatter=formatter,
            suppress_small=True,
            separator="   ",
        )
        return a2s

    def normalize(self):
        """
        return scaled histogram with sum(counts) = 1
        """
        return self / self._counts.sum()

    def rebin(self, nrebin):
        """
        combine `nrebin` bins into 1 bin by summing contents. total
        number of bins for each axis must be divisible by these numbers.
        nbins must be divisible by `nrebin` exactly
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

    def to_poisson_errors(self, alpha=1 - 0.6827):
        """
        set up and down errors to 1 sigma confidence intervals for poisson counts
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

    def svg(self, height=250, aspectratio=1.4, strokewidth=1):
        width = height * aspectratio

        padding = 0.02  # fraction of height or width to keep between edges of plot and svg view size
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
        points.append([padding * width, height * (1 - padding)])

        pathstr = " ".join("{},{}".format(*p) for p in points)

        source = """
        <svg width="{width}" height="{height}" version="1.1" xmlns="http://www.w3.org/2000/svg">
          <rect width="{width}" height="{height}" fill="none" stroke="#000" stroke-width="2" />
          <polyline points="{pathstr}" stroke="#000" fill="#5688C7" stroke-width="{strokewidth}"/>
        </svg>
        """.format(
            width=width, height=height, pathstr=pathstr, strokewidth=strokewidth,
        )
        return source

    def svg_matplotlib(self, **kwargs):
        from io import BytesIO
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 3))
        fig.subplots_adjust(bottom=0.08, right=0.99, top=0.99)
        self.plot(ax=ax, histtype="step", **kwargs)
        # self.plot(ax=ax, **kwargs)
        buf = BytesIO()
        fig.savefig(buf, format="svg")
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        src = "<img src='data:image/svg+xml;base64,{}'/>".format(data)
        return src

    def html_table(self):
        tablerows = []
        nrows = len(self._counts)
        ntoshow = 4  # num of start and end rows to show

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
        # svgsource = self.svg()
        svgsource = self.svg_matplotlib()

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

    def to_json(self):
        def default(obj):
            if hasattr(obj, "__array__"):
                return obj.tolist()
            raise TypeError("Don't know how to serialize object of type", type(obj))

        return json.dumps(self.__dict__, default=default)

    @classmethod
    def from_json(cls, obj):
        obj = json.loads(obj)
        for k in obj:
            if is_listlike(obj[k]):
                obj[k] = np.array(obj[k])
        hnew = cls()
        hnew.__dict__.update(obj)
        return hnew

    @classmethod
    def from_bincounts(cls, counts, bins, errors=None):
        hnew = cls()
        hnew._counts = counts.astype(np.float64)
        hnew._edges = bins
        if errors is not None:
            hnew._errors = errors
        else:
            hnew._errors = hnew._counts * 0
        return hnew

    def plot(self, ax=None, **kwargs):
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()
        kwargs["color"] = kwargs.get("color", self._metadata.get("color"))
        kwargs["label"] = kwargs.get("label", self._metadata.get("label"))
        show_counts = kwargs.pop("show_counts", False)
        show_errors = kwargs.pop("show_errors", False)
        counts_fmt_func = kwargs.pop("counts_fmt_func", "{:g}".format)
        counts_fontsize = kwargs.pop("counts_fontsize", 10)
        counts = self._counts
        edges = self._edges
        yerrs = self._errors
        xerrs = 0.5 * self.bin_widths
        mask = (counts != 0.0) & np.isfinite(counts)
        centers = self.bin_centers

        if show_errors:
            kwargs["fmt"] = kwargs.get("fmt", "o")
            kwargs.pop("histtype", None)
            patches = ax.errorbar(
                centers[mask],
                counts[mask],
                xerr=xerrs[mask],
                yerr=yerrs[mask],
                **kwargs
            )
        else:
            _, _, patches = ax.hist(
                centers[mask], edges, weights=counts[mask], **kwargs
            )

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
        # ax.set_ylim(0,ax.get_ylim()[-1]) # NOTE, user should do this because it messes with logy
        return ax
