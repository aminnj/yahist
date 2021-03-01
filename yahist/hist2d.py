from __future__ import print_function

import numpy as np
import copy
import base64

from .utils import (
    is_listlike,
    is_datelike,
    convert_dates,
    compute_darkness,
    ignore_division_errors,
)

from .hist1d import Hist1D


class Hist2D(Hist1D):
    def _init_numpy(self, obj, **kwargs):
        if len(obj) == 0:
            xs, ys = [], []
        else:
            if "DataFrame" in str(type(obj)):
                self._metadata["pandas_labels"] = obj.columns.tolist()[:2]
                xs = obj.iloc[:, 0].values
                ys = obj.iloc[:, 1].values
            else:
                xs, ys = obj[:, 0], obj[:, 1]

        if ("weights" in kwargs) and (kwargs["weights"] is None):
            kwargs.pop("weights")

        if kwargs.pop("norm", False) or kwargs.pop("density", False):
            raise Exception(
                "Please use the .normalize() method on the histogram object."
            )

        # convert ROOT-like "50,0,10,50,0,10" to [np.linspace(0,10,51), np.linspace(0,10,51)]
        if isinstance(kwargs.get("bins"), str) and (
            kwargs["bins"].count(",") in [2, 5]
        ):
            if kwargs["bins"].count(",") == 2:
                nbinsx, lowx, highx = kwargs["bins"].split(",")
                nbinsy, lowy, highy = nbinsx, lowx, highx
            else:
                nbinsx, lowx, highx, nbinsy, lowy, highy = kwargs["bins"].split(",")
            kwargs["bins"] = [
                np.linspace(float(lowx), float(highx), int(nbinsx) + 1),
                np.linspace(float(lowy), float(highy), int(nbinsy) + 1),
            ]

        if is_datelike(xs):
            xs = convert_dates(xs)
            self._metadata["date_axes"] = ["x"]

        if ("bins" in kwargs) and not isinstance(kwargs["bins"], str):
            bins = kwargs["bins"]
            if is_listlike(bins) and len(bins) == 2:

                if is_datelike(bins[0]):
                    bins[0] = convert_dates(bins[0])
                    self._metadata["date_axes"] = ["x"]

                if kwargs.get("overflow", True):
                    clip_low_x = 0.5 * (bins[0][0] + bins[0][1])
                    clip_high_x = 0.5 * (bins[0][-2] + bins[0][-1])
                    clip_low_y = 0.5 * (bins[1][0] + bins[1][1])
                    clip_high_y = 0.5 * (bins[1][-2] + bins[1][-1])
                    xs = np.clip(xs, clip_low_x, clip_high_x)
                    ys = np.clip(ys, clip_low_y, clip_high_y)

        counts, edgesx, edgesy = _np_histogram2d_wrapper(xs, ys, **kwargs)
        # each row = constant y, lowest y on top
        self._counts = counts.T
        self._edges = edgesx, edgesy
        self._counts = self._counts.astype(np.float64)

        # poisson defaults if not specified
        if self._errors is None:
            if "weights" not in kwargs:
                self._errors = np.sqrt(self._counts)
            else:
                # if weighted entries, need to get sum of sq. weights per bin
                # and sqrt of that is bin error
                kwargs["weights"] = kwargs["weights"] ** 2.0
                counts, _, _ = _np_histogram2d_wrapper(xs, ys, **kwargs)
                self._errors = np.sqrt(counts.T)
        self._errors = self._errors.astype(np.float64)

    def _init_root(self, obj, **kwargs):
        xaxis = obj.GetXaxis()
        yaxis = obj.GetYaxis()
        edges_x = np.array(
            [1.0 * xaxis.GetBinLowEdge(i) for i in range(1, xaxis.GetNbins() + 2)]
        )
        edges_y = np.array(
            [1.0 * yaxis.GetBinLowEdge(i) for i in range(1, yaxis.GetNbins() + 2)]
        )
        counts, errors = [], []
        for iy in range(1, obj.GetNbinsY() + 1):
            counts_y, errors_y = [], []
            for ix in range(1, obj.GetNbinsX() + 1):
                cnt = obj.GetBinContent(ix, iy)
                err = obj.GetBinError(ix, iy)
                counts_y.append(cnt)
                errors_y.append(err)
            counts.append(counts_y[:])
            errors.append(errors_y[:])
        self._counts = np.array(counts, dtype=np.float64)
        self._errors = np.array(errors, dtype=np.float64)
        self._edges = edges_x, edges_y

    def _check_consistency(self, other, raise_exception=True):
        if not (
            np.allclose(self._edges[0], other._edges[0])
            and np.allclose(self._edges[1], other._edges[1])
        ):
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
            and np.allclose(self._edges[0], other.edges[0])
            and np.allclose(self._edges[1], other.edges[1])
            and np.allclose(self._errors, other.errors)
        )

    @property
    def bin_centers(self):
        """
        Returns the centers of bins.

        Returns
        -------
        array
            Bin centers
        """
        xcenters = 0.5 * (self._edges[0][1:] + self._edges[0][:-1])
        ycenters = 0.5 * (self._edges[1][1:] + self._edges[1][:-1])
        return (xcenters, ycenters)

    @property
    def bin_widths(self):
        """
        Returns the widths of bins.

        Returns
        -------
        array
            Bin widths
        """
        xwidths = self._edges[0][1:] - self._edges[0][:-1]
        ywidths = self._edges[1][1:] - self._edges[1][:-1]
        return (xwidths, ywidths)

    @property
    def nbins(self):
        """
        Returns the number of bins

        Returns
        -------
        int
            Number of bins
        """
        return (len(self._edges[0]) - 1, len(self._edges[1]) - 1)

    def _calculate_projection(self, axis, edges):
        hnew = Hist1D()
        hnew._counts = self._counts.sum(axis=axis)
        hnew._errors = np.sqrt((self._errors ** 2).sum(axis=axis))
        hnew._edges = edges
        return hnew

    def projection(self, axis="x"):
        """
        Returns the x-projection of the 2d histogram by
        summing over the y-axis.

        Parameters
        ----------
        axis : str/int (default "x")
            if "x" or 0, return the x-projection (summing over y-axis)
            if "y" or 1, return the y-projection (summing over x-axis)

        Returns
        -------
        Hist1D
        """
        if axis in [0, "x"]:
            iaxis = 0
        elif axis in [1, "y"]:
            iaxis = 1
        else:
            raise Exception("axis parameter must be 'x'/0 or 'y'/1")
        return self._calculate_projection(iaxis, self._edges[iaxis])

    @ignore_division_errors
    def _calculate_profile(self, counts, errors, edges_to_sum, edges):
        centers = 0.5 * (edges_to_sum[:-1] + edges_to_sum[1:])
        num = np.matmul(counts.T, centers)
        den = np.sum(counts, axis=0)
        num_err = np.matmul(errors.T ** 2, centers ** 2) ** 0.5
        den_err = np.sum(errors ** 2, axis=0) ** 0.5
        r_val = num / den
        r_err = ((num_err / den) ** 2 + (den_err * num / den ** 2.0) ** 2.0) ** 0.5
        hnew = Hist1D()
        hnew._counts = r_val
        hnew._errors = r_err
        hnew._edges = edges
        return hnew

    def profile(self, axis="x"):
        """
        Returns the x-profile of the 2d histogram by
        calculating the weighted mean over the y-axis.

        Parameters
        ----------
        axis : str/int (default "x")
            if "x" or 0, return the x-profile (mean over y-axis)
            if "y" or 1, return the y-profile (mean over x-axis)

        Returns
        -------
        Hist1D
        """
        xedges, yedges = self._edges
        if axis in [0, "x"]:
            return self._calculate_profile(self._counts, self._errors, yedges, xedges)
        elif axis in [1, "y"]:
            return self._calculate_profile(
                self._counts.T, self._errors.T, xedges, yedges
            )
        else:
            raise Exception("axis parameter must be 'x'/0 or 'y'/1")

    def transpose(self):
        """
        Returns the transpose of the Hist2D

        Returns
        -------
        Hist2D
        """
        hnew = self.__class__()
        hnew._edges = [self.edges[1], self.edges[0]]
        hnew._errors = self.errors.T
        hnew._counts = self.counts.T
        hnew._metadata = self._metadata.copy()
        return hnew

    def rebin(self, nrebinx, nrebiny=None):
        """
        Combines adjacent bins by summing contents. The total number
        of bins for the x-axis (y-axis) must be exactly divisible by 
        `nrebinx` (`nrebiny`). Based on the method in
        https://stackoverflow.com/questions/44527579/whats-the-best-way-to-downsample-a-numpy-array.

        Parameters
        ----------
        nrebinx : int
            Number of adjacent x-axis bins to combine into one bin.
        nrebiny : int
            Number of adjacent y-axis bins to combine into one bin.

        Returns
        -------
        Hist2D
        """
        ny, nx = self.counts.shape
        by, bx = (nrebinx, nrebinx) if nrebiny is None else (nrebiny, nrebinx)

        if nx % bx != 0:
            raise Exception(
                "This histogram cannot be rebinned since {} is not divisible by {}".format(
                    nx, bx
                )
            )
        if ny % by != 0:
            raise Exception(
                "This histogram cannot be rebinned since {} is not divisible by {}".format(
                    ny, by
                )
            )

        counts = self.counts
        edgesx, edgesy = self.edges
        errors = self.errors

        new_counts = counts.reshape(ny // by, by, nx // bx, bx).sum(axis=(1, 3))
        new_errors = (errors ** 2).reshape(ny // by, by, nx // bx, bx).sum(
            axis=(1, 3)
        ) ** 0.5
        new_edgesx = np.append(edgesx[:-1].reshape(nx // bx, -1).T[0], edgesx[-1])
        new_edgesy = np.append(edgesy[:-1].reshape(ny // by, -1).T[0], edgesy[-1])

        hnew = self.__class__()
        hnew._edges = [new_edgesx, new_edgesy]
        hnew._errors = new_errors
        hnew._counts = new_counts
        hnew._metadata = self._metadata.copy()
        return hnew

    def restrict(self, xlow=None, xhigh=None, ylow=None, yhigh=None):
        """
        Restricts to a contiguous subset of bins with
        bin center values within [[xlow, xhigh], [ylow,yhigh]]. If any limit
        is `None`, the specified direction will be unbounded.

        Parameters
        ----------
        xlow : float (default None)
            Lower x center to keep
        xhigh : float (default None)
            Highest x center to keep
        ylow : float (default None)
            Lower y center to keep
        yhigh : float (default None)
            Highest y center to keep

        Returns
        -------
        Hist2D
        """
        edgesels = [None, None]
        binsels = [None, None]
        for name, axis, low, high in [
            ("x", 0, xlow, xhigh),
            ("y", 1, ylow, yhigh),
        ]:
            centers = self.bin_centers[axis]
            binsel = np.ones_like(centers) > 0.5
            if low is not None:
                binsel &= centers >= low
            if high is not None:
                binsel &= centers <= high
            edgesel = np.concatenate([binsel, [False]])
            if not np.any(edgesel):
                raise Exception(f"No selected bins for {name}-axis. Check the limits.")
            edgesel[np.argwhere(edgesel)[-1][0] + 1] = True

            edgesels[axis] = edgesel
            binsels[axis] = binsel

        h = self.copy()
        h._edges = tuple([h._edges[0][edgesels[0]], h._edges[1][edgesels[1]]])
        h._counts = h._counts[binsels[1], :][:, binsels[0]]
        h._errors = h._errors[binsels[1], :][:, binsels[0]]
        return h

    def cumulative(self, forwardx=True, forwardy=True):
        """
        Turns Hist object into one with cumulative counts.

        Parameters
        ----------
        forwardx : bool, default True
            If true, sum the x-axis from low to high, otherwise high to low
            If None, do not sum along this axis.
        forwardy : bool, default True
            If true, sum the y-axis from low to high, otherwise high to low
            If None, do not sum along this axis.

        Returns
        -------
        Hist2D
        """
        hnew = self.copy()
        # x
        if forwardx is not None:
            directionx = 1 if forwardx else -1
            hnew._counts = hnew._counts[:, ::directionx].cumsum(axis=1)[:, ::directionx]
            hnew._errors = (hnew._errors[:, ::directionx] ** 2.0).cumsum(axis=1)[
                :, ::directionx
            ] ** 0.5
        # y
        if forwardy is not None:
            directiony = 1 if forwardy else -1
            hnew._counts = hnew._counts[::directiony, :].cumsum(axis=0)[::directiony, :]
            hnew._errors = (hnew._errors[::directiony, :] ** 2.0).cumsum(axis=0)[
                ::directiony, :
            ] ** 0.5
        return hnew

    def lookup(self, x, y):
        """
        Convert a specified list of x-values and y-values into corresponding
        bin counts via `np.digitize`

        Parameters
        ----------
        x : array of y-values, or single y-value
        y : array of y-values, or single y-value

        Returns
        -------
        array
        """
        ibins = []
        for axis, vals in enumerate([x, y]):
            low = self.edges[axis][0] + self.bin_widths[axis][0] * 0.5
            low = 0.5 * (self.edges[axis][0] + self.edges[axis][1])
            high = 0.5 * (self.edges[axis][-1] + self.edges[axis][-2])
            vals = np.clip(vals, low, high)
            ibins.append(np.digitize(vals, bins=self.edges[axis]) - 1)
        ibins = tuple(ibins)[::-1]
        return self.counts[ibins]

    def smooth(self, ntimes=3, window=3):
        """
        Returns a smoothed Hist2D via
        convolution with three kernels used by
        https://root.cern.ch/doc/master/TH2_8cxx_source.html#l02600

        Parameters
        ----------
        ntimes : int (default 3)
            Number of times to repeat smoothing
        window : int (default 3)
            Kernel size (1, 3, 5 supported)

        Returns
        -------
        Hist2D
        """
        from scipy.signal import convolve2d

        kernels = {
            1: np.array([[1],]),
            3: np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0],]),
            5: np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 2, 2, 2, 0],
                    [1, 2, 5, 2, 1],
                    [0, 2, 2, 2, 0],
                    [0, 0, 1, 0, 0],
                ]
            ),
        }
        kernel = kernels.get(window)
        if kernel is None:
            raise Exception(
                f"Window/kernel size {window} not supported. Supported sizes: {kernels.keys()}"
            )
        h = self.copy()
        for _ in range(ntimes):
            h._counts = convolve2d(h._counts, kernel, mode="same") / kernel.sum()
            h._errors = (
                convolve2d(h._errors ** 2.0, kernel, mode="same") ** 0.5 / kernel.sum()
            )
        return h

    def sample(self, size=1e5):
        """
        Returns a 2-column array of random samples according
        to a discrete pdf from this histogram.

        >>> h1 = Hist2D.from_random()
        >>> h2 = Hist2D(h1.sample(100), bins=h1.edges)

        Parameters
        ----------
        size : int/float, 1e5
            Number of random values to sample

        Returns
        -------
        array
        """
        counts = self.counts
        xcenters, ycenters = self.bin_centers

        # triplets of (x bin center, y bin center, count)
        xyz = np.c_[
            np.tile(xcenters, len(ycenters)),
            np.repeat(ycenters, len(xcenters)),
            counts.flatten(),
        ][counts.flatten() != 0]

        idx = np.arange(len(xyz))
        probs = xyz[:, 2] / xyz[:, 2].sum()
        sampled_idx = np.random.choice(idx, size=int(size), p=probs)
        xy = xyz[sampled_idx][:, [0, 1]]

        return xy

    def svg_fast(self, height=250, aspectratio=1.4, interactive=True):
        """
        Return HTML svg tag with bare-bones version of histogram
        (no ticks, labels).

        Parameters
        ----------
        height : int, default 250
            Height of plot in pixels
        aspectratio : float, default 1.4
            Aspect ratio of plot
        interactive : bool, default True
            Whether to display bin contents on mouse hover.

        Returns
        -------
        str
        """
        width = height * aspectratio

        template = """
        <svg width="{svg_width}" height="{svg_height}" version="1.1" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">

          {{content}}

          <rect width="{plot_width}" height="{plot_height}" fill="none" stroke="#000" stroke-width="2" />
        </svg>
        """.format(
            plot_width=width,
            svg_width=width,
            plot_height=height,
            svg_height=height + (30 if interactive else 0),
        )

        if interactive:
            template += """
            <style>
                svg rect.bar:hover {{ stroke: red; }}
                svg text {{ display: none; }}
                svg g:hover text {{ display: block; }}
            </style>
            """

        ex = self.edges[0]
        ey = self.edges[1]
        counts = self.counts

        # transform edges into svg coordinates, and counts into [0..255] for color
        t_ex = width * (ex - ex.min()) / ex.ptp()
        t_ey = height * (1.0 - (ey - ey.min()) / ey.ptp())
        t_counts = counts
        t_counts_norm = 100 * (1 - (counts - counts.min()) / counts.ptp())

        X, Y = np.meshgrid(t_ex, t_ey)

        mat = np.c_[
            X[:-1, :-1].flatten(),
            X[1:, 1:].flatten(),
            Y[1:, 1:].flatten(),
            Y[:-1, :-1].flatten(),
            t_counts.flatten(),
            t_counts_norm.flatten(),
        ]

        content = []
        if interactive:
            rect_str = (
                """<g><rect class="bar" width="{w:g}" height="{h:g}" x="{x:g}" y="{y:g}" fill="hsl(0,0%%,{c:.2f}%%)"/>"""
                """<text text-anchor="middle" x="%.0f" y="%.0f">x={xmid:g}, y={ymid:g}, z={z:g}</text></g>"""
            ) % (width / 2, height + 30 / 2)
        else:
            rect_str = """<rect width="{w:g}" height="{h:g}" x="{x:g}" y="{y:g}" fill="hsl(0,0%,{c:.2f}%)"/>"""
        for x1, x2, y1, y2, z, c in mat[mat[:, -2] > 0]:
            w = x2 - x1
            h = y2 - y1
            xmid = 0.5 * (x1 + x2)
            ymid = 0.5 * (y1 + y2)
            content.append(
                rect_str.format(w=w, h=h, x=x1, y=y1, c=c, z=z, xmid=xmid, ymid=ymid)
            )

        source = template.format(content="\n".join(content))
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
        fig.subplots_adjust(left=0.15, bottom=0.15, right=0.96, top=0.96)
        self.plot(ax=ax, **kwargs)
        buf = BytesIO()
        fig.savefig(buf, format="svg")
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        src = "<img src='data:image/svg+xml;base64,{}'/>".format(data)
        return src

    def canvas(self, height=250, aspectratio=1.4):
        """
        Return HTML5 canvas tag similar to `self.svg()`.

        Parameters
        ----------
        height : int, default 250
            Height of plot in pixels
        aspectratio : float, default 1.4
            Aspect ratio of plot

        Returns
        -------
        str
        """
        width = height * aspectratio

        # make sure the canvas id is unique to avoid clobbering old ones
        import random

        chash = "%010x" % random.randrange(16 ** 10)

        template = """
        <script type="text/javascript">
            function S(x, y, w, h, fill) {{
                this.x = x;
                this.y = y;
                this.w = w;
                this.h = h;
                this.fill = fill;
            }}
            var elem = document.getElementById('myCanvas_%s');
            if (elem.getContext) {{
                var rects = [];
                {content}
                context = elem.getContext('2d');
                for (var rect of rects) {{
                    context.fillStyle = rect.fill;
                    context.fillRect(rect.x, rect.y, rect.w, rect.h);
                }}
            }}
        </script>

        <canvas id="myCanvas_%s" width="%.0f" height="%.0f" style="border:1px solid #000000;"></canvas>
        """ % (
            chash,
            chash,
            width,
            height,
        )

        ex = self.edges[0]
        ey = self.edges[1]
        counts = self.counts

        # transform edges into svg coordinates, and counts into [0..255] for color
        t_ex = width * (ex - ex.min()) / ex.ptp()
        t_ey = height * (1.0 - (ey - ey.min()) / ey.ptp())
        t_counts = counts
        t_counts_norm = (100 * (1 - (counts - counts.min()) / counts.ptp())).astype(int)

        X, Y = np.meshgrid(t_ex, t_ey)

        mat = np.c_[
            X[:-1, :-1].flatten(),
            X[1:, 1:].flatten(),
            Y[1:, 1:].flatten(),
            Y[:-1, :-1].flatten(),
            t_counts.flatten(),
            t_counts_norm.flatten(),
        ]

        content = []
        rect_str = """rects.push(new S(%g, %g, %g, %g, "hsl(0,0%%,%g%%)"));"""
        for x1, x2, y1, y2, z, c in mat[mat[:, -2] > 0]:
            w = x2 - x1
            h = y2 - y1
            content.append(rect_str % (x1, y1, w, h, c))

        source = template.format(content="\n".join(content))
        return source

    def html_table(self):
        """
        Return dummy HTML table tag.

        Returns
        -------
        str
        """
        return ""

    def _repr_html_(self):
        tablestr = self.html_table()
        imgsource = self.svg()

        source = """
        <div style="max-height:1000px;max-width:1500px;overflow:auto">
        <b>total count</b>: {count}, <b>metadata</b>: {metadata}<br>
        <div style="display:flex;">
            {imgsource}
        </div>
        """.format(
            count=self._counts.sum(),
            metadata=self._metadata,
            imgsource=imgsource,
            tablestr=tablestr,
        )
        return source

    def plot(self, ax=None, fig=None, **kwargs):
        """
        Plot this histogram object using matplotlib's `hist`
        function, or `errorbar`.

        Parameters
        ----------
        ax : matplotlib AxesSubplot object, default None
            matplotlib AxesSubplot object. Created if `None`.
        fig : matplotlib Figure object, default None
            matplotlib Figure object. Created if `None`.
        counts
            Alias for `show_counts`
        counts_fmt_func : function, default "{:3g}".format
            Function used to format count labels
        counts_fontsize
            Font size of count labels
        colorbar : bool, default True
            Show colorbar
        equidistant : str ("", "x", "y", "xy"), default ""
            If not an empty string, make bins equally-spaced in the x-axis ("x"),
            y-axis ("y"), or both ("xy").
        hide_empty : bool, default True
            Don't draw empty bins (content==0)
        interactive : bool, default False
            Use plotly to make an interactive plot
        logz : bool, default False
            Use logscale for z-axis
        show_counts : bool, default False
            Show count labels for each bin
        show_errors : bool, default False
            Show error bars
        return_self : bool, default False
            If true, return self (Hist2D object)
        **kwargs
            Parameters to be passed to matplotlib 
            `pcolorfast` function.


        Returns
        -------
        Hist2D (self) if `return_self` is True, otherwise
        (pcolorfast output, matplotlib AxesSubplot object)
        """

        if kwargs.pop("interactive", False):
            return self.plot_plotly(**kwargs)

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()

        counts = self._counts
        xedges, yedges = self._edges

        show_counts = kwargs.pop("show_counts", kwargs.pop("counts", False))
        colorbar = kwargs.pop("colorbar", True)
        hide_empty = kwargs.pop("hide_empty", True)
        counts_fmt_func = kwargs.pop("counts_fmt_func", "{:3g}".format)
        counts_fontsize = kwargs.pop("counts_fontsize", 12)
        logz = kwargs.pop("logz", False)
        equidistant = kwargs.pop("equidistant", "")
        return_self = kwargs.pop("return_self", False)

        if logz:
            kwargs["norm"] = LogNorm()

        countsdraw = counts
        if hide_empty:
            countsdraw = np.array(counts)
            countsdraw[countsdraw == 0] = np.nan

        if equidistant:
            c = ax.imshow(
                countsdraw[::-1, ::1],
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                interpolation="none",
                aspect="auto",
                **kwargs,
            )
            if "x" in equidistant:
                ax.xaxis.set_ticks(np.linspace(xedges[0], xedges[-1], len(xedges)))
                ax.xaxis.set_ticklabels(xedges)
            if "y" in equidistant:
                ax.yaxis.set_ticks(np.linspace(yedges[0], yedges[-1], len(yedges)))
                ax.yaxis.set_ticklabels(yedges)
        else:
            c = ax.pcolorfast(xedges, yedges, countsdraw, **kwargs)

        if colorbar:
            cbar = fig.colorbar(c, ax=ax)

        if "pandas_labels" in self.metadata:
            xlabel, ylabel = self.metadata["pandas_labels"]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        if "date_axes" in self.metadata:
            import matplotlib.dates

            locator = matplotlib.dates.AutoDateLocator()
            formatter = matplotlib.dates.ConciseDateFormatter(locator)
            which_axes = self.metadata["date_axes"]
            if "x" in which_axes:
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

        if show_counts:
            xcenters, ycenters = self.bin_centers
            if equidistant:
                if "x" in equidistant:
                    xcenters = np.linspace(xedges[0], xedges[-1], len(xedges))
                    xcenters = 0.5 * (xcenters[:-1] + xcenters[1:])
                if "y" in equidistant:
                    ycenters = np.linspace(yedges[0], yedges[-1], len(yedges))
                    ycenters = 0.5 * (ycenters[:-1] + ycenters[1:])
            xyz = np.c_[
                np.tile(xcenters, len(ycenters)),
                np.repeat(ycenters, len(xcenters)),
                counts.flatten(),
            ][counts.flatten() != 0]

            colors = np.zeros((len(xyz), 3))
            r, g, b, a = c.to_rgba(xyz[:, 2]).T
            colors[compute_darkness(r, g, b, a) > 0.45] = 1

            for (x, y, z), color in zip(xyz, colors):
                ax.text(
                    x,
                    y,
                    counts_fmt_func(z),
                    color=color,
                    ha="center",
                    va="center",
                    fontsize=counts_fontsize,
                    wrap=True,
                )

        if return_self:
            return self
        else:
            return c, ax

    def plot_plotly(self, **kwargs):
        import plotly.graph_objects as go

        xcenters, ycenters = self.bin_centers
        counts = self.counts
        xyz = np.c_[
            np.tile(xcenters, len(ycenters)),
            np.repeat(ycenters, len(xcenters)),
            counts.flatten(),
        ][counts.flatten() != 0]

        cbtitle = None
        x, y, z = xyz.T
        if kwargs.get("logz"):
            z = np.log10(z)
            cbtitle = "log10(bin)"

        fig = go.Figure(
            go.Histogram2d(
                x=x,
                y=y,
                z=z,
                histfunc="sum",
                xbins=dict(
                    start=self.edges[0][0],
                    end=self.edges[0][-1],
                    size=self.bin_widths[0][0],
                ),
                ybins=dict(
                    start=self.edges[1][0],
                    end=self.edges[1][-1],
                    size=self.bin_widths[1][0],
                ),
                colorscale=[
                    [0.00, "rgba(0,0,0,0)"],
                    [1e-6, "rgb(68, 1, 84)"],
                    [0.25, "rgb(59, 82, 139)"],
                    [0.50, "rgb(33, 145, 140)"],
                    [0.75, "rgb(94, 201, 98)"],
                    [1.00, "rgb(253, 231, 37)"],
                ],
                colorbar=dict(thicknessmode="fraction", thickness=0.04, title=cbtitle),
            ),
        )
        fig.update_layout(
            height=300,
            width=400,
            template="simple_white",
            font_family="Arial",
            xaxis=dict(mirror=True,),
            yaxis=dict(mirror=True,),
            margin=dict(l=10, r=10, b=10, t=30, pad=0,),
        )
        return fig


def _compute_bin_1d_uniform(x, bins, overflow=False):
    n = bins.shape[0] - 1
    b_min = bins[0]
    b_max = bins[-1]
    if overflow:
        if x > b_max:
            return n - 1
        elif x < b_min:
            return 0
    ibin = int(n * (x - b_min) / (b_max - b_min))
    if x < b_min or x > b_max:
        return -1
    else:
        return ibin


def _numba_histogram2d(ax, ay, bins_x, bins_y, weights=None, overflow=False):
    db_x = np.ediff1d(bins_x)
    db_y = np.ediff1d(bins_y)
    is_uniform_binning_x = np.all(db_x - db_x[0] < 1e-6)
    is_uniform_binning_y = np.all(db_y - db_y[0] < 1e-6)
    hist = np.zeros((len(bins_x) - 1, len(bins_y) - 1), dtype=np.float64)
    ax = ax.flat
    ay = ay.flat
    b_min_x = bins_x[0]
    b_max_x = bins_x[-1]
    n_x = bins_x.shape[0] - 1
    b_min_y = bins_y[0]
    b_max_y = bins_y[-1]
    n_y = bins_y.shape[0] - 1
    if weights is None:
        weights = np.ones(len(ax), dtype=np.float64)
    if is_uniform_binning_x and is_uniform_binning_y:
        for i in range(len(ax)):
            ibin_x = _compute_bin_1d_uniform(ax[i], bins_x, overflow=overflow)
            ibin_y = _compute_bin_1d_uniform(ay[i], bins_y, overflow=overflow)
            if ibin_x >= 0 and ibin_y >= 0:
                hist[ibin_x, ibin_y] += weights[i]
    else:
        ibins_x = np.searchsorted(bins_x, ax, side="left")
        ibins_y = np.searchsorted(bins_y, ay, side="left")
        for i in range(len(ax)):
            ibin_x = ibins_x[i]
            ibin_y = ibins_y[i]
            if overflow:
                if ibin_x == n_x + 1:
                    ibin_x = n_x
                elif ibin_x == 0:
                    ibin_x = 1
                if ibin_y == n_y + 1:
                    ibin_y = n_y
                elif ibin_y == 0:
                    ibin_y = 1
            if ibin_x >= 1 and ibin_y >= 1 and ibin_x <= n_x and ibin_y <= n_y:
                hist[ibin_x - 1, ibin_y - 1] += weights[i]
    return hist, bins_x, bins_y


_numba_histogram2d._wrapped = False


def _np_histogram2d_wrapper(x, y, overflow=True, **kwargs):
    # Yuck, globals. This is so we don't incur the cost of importing numba (~1sec)
    # if we never end up using it.
    global _compute_bin_1d_uniform, _numba_histogram2d

    if kwargs.pop("allow_numba", True) and (len(x) > 1e5):

        HAS_NUMBA = False
        try:
            import numba

            HAS_NUMBA = True

            if not _numba_histogram2d._wrapped:
                jitfunc = numba.jit(nopython=True, nogil=True)
                _compute_bin_1d_uniform = jitfunc(_compute_bin_1d_uniform)
                _numba_histogram2d = jitfunc(_numba_histogram2d)
                _numba_histogram2d._wrapped = True
        except:
            pass

        if HAS_NUMBA:
            bins = kwargs.get("bins", None)
            # check if `bins` is a 2 element list of lists (x bins, y bins)
            if is_listlike(bins) and (len(bins) == 2) and (is_listlike(bins[0])):
                bins_x, bins_y = bins
                weights = kwargs.get("weights", None)
                return _numba_histogram2d(
                    x, y, bins_x, bins_y, weights=weights, overflow=overflow
                )
            # or if single dimension
            if is_listlike(bins) and (np.ndim(bins) == 1):
                bins_x = bins_y = bins
                weights = kwargs.get("weights", None)
                return _numba_histogram2d(
                    x, y, bins_x, bins_y, weights=weights, overflow=overflow
                )

    return np.histogram2d(x, y, **kwargs)
