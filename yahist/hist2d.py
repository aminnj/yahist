from __future__ import print_function

import numpy as np
import copy
import base64

from .utils import is_listlike, compute_darkness

from .hist1d import Hist1D


class Hist2D(Hist1D):
    def _init_numpy(self, obj, **kwargs):
        if len(obj) == 0:
            xs, ys = [], []
        else:
            # FIXME, should take a tuple of xs, ys since obj can be arbitrary
            xs, ys = obj[:, 0], obj[:, 1]

        if (
            kwargs.pop("overflow", True)
            and ("bins" in kwargs)
            and not isinstance(kwargs["bins"], str)
        ):
            bins = kwargs["bins"]
            if is_listlike(bins) and len(bins) == 2:
                clip_low_x = 0.5 * (bins[0][0] + bins[0][1])
                clip_high_x = 0.5 * (bins[0][-2] + bins[0][-1])
                clip_low_y = 0.5 * (bins[1][0] + bins[1][1])
                clip_high_y = 0.5 * (bins[1][-2] + bins[1][-1])
                xs = np.clip(xs, clip_low_x, clip_high_x)
                ys = np.clip(ys, clip_low_y, clip_high_y)

        counts, edgesx, edgesy = np.histogram2d(xs, ys, **kwargs)
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
                counts, _, _ = np.histogram2d(xs, ys, **kwargs)
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
        if (len(self._edges[0]) != len(other._edges[0])) or (
            len(self._edges[1]) != len(other._edges[1])
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
        return (len(self._edges[0]) - 1, len(self._edges[1]) -1)

    def _calculate_projection(self, axis, edges):
        hnew = Hist1D()
        hnew._counts = self._counts.sum(axis=axis)
        hnew._errors = np.sqrt((self._errors ** 2).sum(axis=axis))
        hnew._edges = edges
        return hnew

    def projection(self, axis):
        """
        Returns the x-projection of the 2d histogram by
        summing over the y-axis.

        Parameters
        ----------
        axis : str
            if "x", return the x-projection (summing over y-axis)
            if "y", return the y-projection (summing over x-axis)

        Returns
        -------
        Hist1D
        """
        if axis == "x":
            iaxis = 0
        elif axis == "y":
            iaxis = 1
        else:
            raise Exception("axis parameter must be 'x' or 'y'")
        return self._calculate_projection(iaxis, self._edges[iaxis])

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

    def profile(self, axis):
        """
        Returns the x-profile of the 2d histogram by
        calculating the weighted mean over the y-axis.

        Parameters
        ----------
        axis : str
            if "x", return the x-profile (mean over y-axis)
            if "y", return the y-profile (mean over x-axis)

        Returns
        -------
        Hist1D
        """
        xedges, yedges = self._edges
        if axis == "x":
            return self._calculate_profile(self._counts, self._errors, yedges, xedges)
        elif axis == "y":
            return self._calculate_profile(
                self._counts.T, self._errors.T, xedges, yedges
            )
        else:
            raise Exception("axis parameter must be 'x' or 'y'")

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
        by, bx = (nrebinx, nrebinx) if nrebiny is None else (nrebinx, nrebiny)

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

    def svg(self, height=250, aspectratio=1.4, interactive=True):
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

    def svg_matplotlib(self, **kwargs):
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
        fig.subplots_adjust(bottom=0.08, right=0.95, top=0.96)
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
        imgsource = self.svg_matplotlib()

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
        counts_fmt_func : function, default "{:3g}".format
            Function used to format count labels
        counts_fontsize
            Font size of count labels
        logz : bool, default False
            Use logscale for z-axis
        show_counts : bool, default False
            Show count labels for each bin
        show_errors : bool, default False
            Show error bars
        **kwargs
            Parameters to be passed to matplotlib 
            `pcolorfast` function.


        Returns
        -------
        (pcolorfast output, matplotlib AxesSubplot object)
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()

        counts = self._counts
        xedges, yedges = self._edges

        show_counts = kwargs.pop("show_counts", False)
        counts_fmt_func = kwargs.pop("counts_fmt_func", "{:3g}".format)
        counts_fontsize = kwargs.pop("counts_fontsize", 12)
        logz = kwargs.pop("logz", False)

        if logz:
            kwargs["norm"] = LogNorm()

        c = ax.pcolorfast(xedges, yedges, counts, **kwargs)
        cbar = fig.colorbar(c, ax=ax)

        if show_counts:
            xcenters, ycenters = self.bin_centers
            xyz = np.c_[
                np.tile(xcenters, len(ycenters)),
                np.repeat(ycenters, len(xcenters)),
                counts.flatten(),
            ][counts.flatten() != 0]

            r, g, b, a = cbar.mappable.to_rgba(xyz[:, 2]).T
            colors = np.zeros((len(xyz), 3))
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

        return c, ax
