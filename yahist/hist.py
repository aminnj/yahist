from __future__ import print_function

import sys
import numpy as np
import copy
import json

from .utils import (
        is_listlike,
        clopper_pearson_error,
        poisson_errors,
        ignore_division_errors,
        )


PY2 = True
if sys.version_info[0] >= 3:
    PY2 = False

class Hist1D(object):

    def __init__(self, obj=[], **kwargs):
        self._counts, self._edges, self._errors = None, None, None
        self._errors_up, self._errors_down = None, None # used when dividing with binomial errors
        self._metadata = {}
        kwargs = self._extract_metadata(**kwargs)
        if is_listlike(obj):
            self._init_numpy(obj,**kwargs)
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
        if kwargs.pop("overflow",True) and ("bins" in kwargs) and not isinstance(kwargs["bins"],str):
            bins = kwargs["bins"]
            clip_low = 0.5*(bins[0] + bins[1])
            clip_high = 0.5*(bins[-2] + bins[-1])
            obj = np.clip(obj,clip_low,clip_high)
        self._counts, self._edges = np.histogram(obj,**kwargs)
        self._counts = self._counts.astype(np.float64)

        # poisson defaults if not specified
        if self._errors is None:
            if "weights" not in kwargs:
                self._errors = np.sqrt(self._counts)
            else:
                # if weighted entries, need to get sum of sq. weights per bin
                # and sqrt of that is bin error
                kwargs["weights"] = kwargs["weights"]**2.
                counts, _ = np.histogram(obj,**kwargs)
                self._errors = np.sqrt(counts)
        self._errors = self._errors.astype(np.float64)

    def _extract_metadata(self, **kwargs):
        for k in ["color", "label"]:
            if k in kwargs:
                self._metadata[k] = kwargs.pop(k)
        return kwargs

    @property
    def errors(self): return self._errors

    @property
    def errors_up(self): return self._errors_up

    @property
    def errors_down(self): return self._errors_down

    @property
    def counts(self): return self._counts

    @property
    def edges(self): return self._edges

    @property
    def bin_centers(self):
        return 0.5*(self._edges[1:]+self._edges[:-1])

    @property
    def bin_widths(self):
        return self._edges[1:]-self._edges[:-1]

    @property
    def integral(self):
        return self.counts.sum()

    @property
    def integral_error(self):
        return (self._errors**2.0).sum()**0.5

    def _fix_nan(self):
        for x in [self._counts,self._errors,
                self._errors_up,self._errors_down]:
            if x is not None:
                np.nan_to_num(x,copy=False)

    def _check_consistency(self, other, raise_exception=True):
        if len(self._edges) != len(other._edges):
            if raise_exception:
                raise Exception("These histograms cannot be combined due to different binning")
            else:
                return False
        return True

    def __eq__(self, other):
        if not self._check_consistency(other, raise_exception=False): return False
        return (
                np.allclose(self._counts, other.counts) and
                np.allclose(self._edges, other.edges) and
                np.allclose(self._errors, other.errors) and
                ((self._errors_up is not None) or np.allclose(self._errors_up, other.errors_up)) and
                ((self._errors_down is not None) or np.allclose(self._errors_down, other.errors_down))
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
        hnew._errors = (self._errors**2. + other._errors**2.)**0.5
        hnew._edges = self._edges
        hnew._metadata = self._metadata.copy()
        return hnew

    __radd__ = __add__

    def __sub__(self, other):
        self._check_consistency(other)
        hnew = self.__class__()
        hnew._counts = self._counts - other._counts
        hnew._errors = (self._errors**2. + other._errors**2.)**0.5
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
                    (self._errors/other._counts)**2.0 +
                    (other._errors*self._counts/(other._counts)**2.0)**2.0
                    )**0.5
            if self._errors_up is not None:
                hnew._errors_up = (
                        (self._errors_up/other._counts)**2.0 +
                        (other._errors*self._counts/(other._counts)**2.0)**2.0
                        )**0.5
                hnew._errors_down = (
                        (self._errors_down/other._counts)**2.0 +
                        (other._errors*self._counts/(other._counts)**2.0)**2.0
                        )**0.5
        else:
            bothzero = (self._counts==0) & (other._counts==0)
            hnew._errors_down, hnew._errors_up = clopper_pearson_error(self._counts,other._counts)
            hnew._counts = self._counts/other._counts
            # these are actually the positions for down and up, but we want the errors
            # wrt to the central value
            hnew._errors_up = hnew._errors_up - hnew._counts
            hnew._errors_down = hnew._counts - hnew._errors_down
            hnew._errors = 0.5*(hnew._errors_down + hnew._errors_up) # nominal errors are avg of up and down
            # For consistency with TEfficiency, up error is 1 if we have 0/0
            hnew._errors_up[bothzero] = 1.
        # hnew._fix_nan()
        return hnew

    def __div__(self, other):
        if type(other) in [float,int,np.float64]:
            return self.__mul__(1./other)
        elif is_listlike(other):
            # Divide histogram by array (counts) assuming errors are 0
            other = np.array(other)
            if len(other) != len(self._counts):
                raise Exception("Cannot divide due to different binning")
            hnew = self.__class__()
            hnew._edges = self._edges
            hnew._counts = other
            hnew._errors = 0.*hnew._counts
            return self.divide(hnew)
        else:
            return self.divide(other)

    __truediv__ = __div__


    def __mul__(self, fact):
        if type(fact) in [float,int,np.float64]:
            hnew = self._copy()
            hnew._counts *= fact
            hnew._errors *= fact
            return hnew
        else:
            raise Exception("Can't multiply histogram by non-scalar")

    __rmul__ = __mul__

    def __pow__(self, expo):
        if type(expo) in [float,int,np.float64]:
            hnew = self._copy()
            hnew._counts = hnew._counts ** expo
            hnew._errors *= hnew._counts**(expo-1) * expo
            return hnew
        else:
            raise Exception("Can't exponentiate histogram by non-scalar")

    def __repr__(self):
        sep = u"\u00B1"
        if PY2: sep = sep.encode("utf-8")
        # trick: want to use numpy's smart formatting (truncating,...) of arrays
        # so we convert value,error into a complex number and format that 1D array :)
        prec = np.get_printoptions()["precision"]
        if prec == 8: prec = 3
        formatter = {"complex_kind": lambda x:"%5.{}f {} %4.{}f".format(prec,sep,prec) % (np.real(x),np.imag(x))}
        # formatter = {"complex_kind": lambda x:("{:g} %s {:g}" % (sep)).format(np.real(x),np.imag(x))}
        a2s = np.array2string(self._counts+self._errors*1j,formatter=formatter, suppress_small=True, separator="   ")
        return a2s

    def normalize(self):
        """
        return scaled histogram with sum(counts) = 1
        """
        return self / self._counts.sum()

    def rebin(self, nrebin):
        """
        combine `nrebin` bins into 1 bin, so
        nbins must be divisible by `nrebin` exactly
        """
        if (len(self._edges)-1) % nrebin != 0:
            raise Exception("This histogram cannot be rebinned since {} is not divisible by {}".format(len(self.edges)-1,nrebin))
        errors2 = self._errors**2.
        # TODO can be more numpythonic, but I was lazy 
        new_counts = [sum(self._counts[i*nrebin:(i+1)*nrebin]) for i in range(0, len(self._edges)//nrebin)]
        new_errors2 = [sum(errors2[i*nrebin:(i+1)*nrebin]) for i in range(0, len(self._edges)//nrebin)]
        new_edges = self._edges[::nrebin]
        hnew = self.__class__()
        hnew._edges = np.array(new_edges)
        hnew._errors = np.array(new_errors2)**0.5
        hnew._counts = np.array(new_counts)
        hnew._metadata = self._metadata.copy()
        return hnew

    def to_poisson_errors(self,alpha=1-0.6827):
        """
        set up and down errors to 1 sigma confidence intervals for poisson counts
        """
        lows, highs = poisson_errors(self._counts,alpha=alpha)
        hnew = self.__class__()
        hnew._counts = np.array(self._counts)
        hnew._edges = np.array(self._edges)
        hnew._errors = np.array(self._errors)
        hnew._errors_up = np.array(highs-self._counts)
        hnew._errors_down = np.array(self._counts-lows)
        hnew._metadata = self._metadata.copy()
        return hnew


    def svg(self, height=250, aspectratio=1.4, strokewidth=1):
        width = height*aspectratio

        padding = 0.02 # fraction of height or width to keep between edges of plot and svg view size
        safecounts = np.array(self._counts)
        safecounts[~np.isfinite(safecounts)] = 0.

        # map 0,max -> height-padding*height,0+padding*height
        ys = height*((2*padding-1)/safecounts.max()*safecounts + (1-padding))
        # map min,max -> padding*width,width-padding*width
        xs = width*((1-2*padding)/(self._edges.max()-self._edges.min())*(self._edges-self._edges.min())+ padding)

        points = []
        points.append([padding*width,height*(1-padding)])
        for i in range(len(xs)-1):
            points.append([xs[i],ys[i]])
            points.append([xs[i+1],ys[i]])
        points.append([width*(1-padding),height*(1-padding)])
        points.append([padding*width,height*(1-padding)])

        pathstr = " ".join("{},{}".format(*p) for p in points)

        template = """
        <svg width="{width}" height="{height}" version="1.1" xmlns="http://www.w3.org/2000/svg">
          <rect width="{width}" height="{height}" fill="transparent" stroke="#000" stroke-width="2" />
          <polyline points="{pathstr}" stroke="#000" fill="#5688C7" stroke-width="{strokewidth}"/>
        </svg>
        """.format(
           width=width,height=height,
           pathstr=pathstr,strokewidth=strokewidth,
        )
        return template

    def html_table(self):
        tablerows = []
        nrows = len(self._counts)
        ntohide = 4 # num to hide on each side
        def format_row(low,high,count,error):
            return "<tr><td>({:g},{:g})</td><td>{:g} \u00B1 {:g}</td></tr>".format(low,high,count,error)
        # NOTE, can be optimized: we don't need to convert every row if we will hide some later
        for lhce in zip(self._edges[:-1],self._edges[1:],self._counts,self._errors):
            tablerows.append(format_row(*lhce))
        if nrows < ntohide*2 + 2: # we don't ever want to hide just 1 row
            tablestr = " ".join(tablerows)
        else:
            nhidden = nrows-ntohide*2 # number hidden in the middle
            tablerows = (tablerows[:ntohide]+
                         ["<tr><td colspan='2'><center>[{} rows hidden]</center></td></tr>".format(nhidden)]+
                         tablerows[-ntohide:])
            tablestr = "\n".join(tablerows)
        return tablestr

    def _repr_html_(self):
        tablestr = self.html_table()
        svgsource = self.svg()

        template = """
        <div style="max-height:1000px;max-width:1500px;overflow:auto">
        <b>total count</b>: {count}, <b>metadata</b>: {metadata}<br>
        <div style="display:flex;">
            <div style="display:inline;">
                <table style='border:1px solid black;'">
                    <thead><tr><th>bin</th><th>content</th></tr></thead>
                    {tablestr}
                </table>
            </div>
            <div style="display:inline; margin: auto 2%;">
                {svgsource}
            </div>
            </div>
        </div>
        """.format(
            count=self._counts.sum(),metadata=self._metadata,
                    svgsource=svgsource,
                   tablestr=tablestr,
        )
        return template

    def to_json(self):
        def default(obj):
            if hasattr(obj,"__array__"): 
                return obj.tolist()
            raise TypeError("Don't know how to serialize object of type",type(obj))
        return json.dumps(self.__dict__,default=default)

    @classmethod
    def from_json(cls, obj):
        obj = json.loads(obj)
        for k in obj:
            if is_listlike(obj[k]):
                obj[k] = np.array(obj[k])
        hnew = cls()
        hnew.__dict__.update(obj)
        return hnew

    def plot(self,ax=None,**kwargs):
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        kwargs["color"] = kwargs.get("color",self._metadata.get("color"))
        kwargs["label"] = kwargs.get("label",self._metadata.get("label"))
        show_counts = kwargs.pop("show_counts",False)
        show_errors = kwargs.pop("show_errors",False)
        counts = self._counts
        edges = self._edges
        yerrs = self._errors
        xerrs = 0.5*self.bin_widths
        mask = (counts != 0.) & np.isfinite(counts)
        centers = self.bin_centers

        if show_errors:
            kwargs["fmt"] = kwargs.get("fmt","o")
            patches = ax.errorbar(centers[mask],counts[mask],xerr=xerrs[mask],yerr=yerrs[mask],**kwargs)
        else:
            _, _, patches = ax.hist(centers[mask],edges,weights=counts[mask],**kwargs)

        if show_counts:
            patch = patches[0]
            color = None
            if hasattr(patch,"get_color"): color = patch.get_color()
            elif hasattr(patch,"get_facecolor"): color = patch.get_facecolor()
            xtodraw = centers[mask]
            ytexts = counts[mask]
            if show_errors:
                ytodraw = counts[mask]+yerrs[mask]
            else:
                ytodraw = ytexts
            for x,y,ytext in zip(xtodraw,ytodraw,ytexts):
                ax.text(x,y,"{:g}".format(ytext), horizontalalignment="center",verticalalignment="bottom", fontsize=10, color=color)
        ax.set_ylim(0,ax.get_ylim()[-1])
        return ax


class Hist2D(Hist1D):

    def _init_numpy(self, obj, **kwargs):
        if len(obj) == 0:
            xs, ys = [],[]
        else:
            xs, ys = obj[:,0], obj[:,1]
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
                kwargs["weights"] = kwargs["weights"]**2.
                counts, _, _ = np.histogram2d(obj[:,0],obj[:,1],**kwargs)
                self._errors = np.sqrt(counts.T)
        self._errors = self._errors.astype(np.float64)

    def _check_consistency(self, other, raise_exception=True):
        if (len(self._edges[0]) != len(other._edges[0])) \
                or (len(self._edges[1]) != len(other._edges[1])):
            if raise_exception:
                raise Exception("These histograms cannot be combined due to different binning")
            else:
                return False
        return True

    def __eq__(self, other):
        if not self._check_consistency(other, raise_exception=False): return False
        return (
                np.allclose(self._counts, other.counts) and
                np.allclose(self._edges[0], other.edges[0]) and
                np.allclose(self._edges[1], other.edges[1]) and
                np.allclose(self._errors, other.errors)
                )

    def bin_centers(self):
        xcenters = 0.5*(self._edges[0][1:]+self._edges[0][:-1])
        ycenters = 0.5*(self._edges[1][1:]+self._edges[1][:-1])
        return (xcenters,ycenters)

    def bin_widths(self):
        xwidths = self._edges[0][1:]-self._edges[0][:-1]
        ywidths = self._edges[1][1:]-self._edges[1][:-1]
        return (xwidths,ywidths)

    def _calculate_projection(self, axis, edges):
        hnew = Hist1D()
        hnew._counts = self._counts.sum(axis=axis)
        hnew._errors = np.sqrt((self._errors**2).sum(axis=axis))
        hnew._edges = edges
        return hnew


    def x_projection(self):
        return self._calculate_projection(0, self._edges[0])

    def y_projection(self):
        return self._calculate_projection(1, self._edges[1])

    def _calculate_profile(self, counts, errors, edges_to_sum, edges):
        centers = 0.5*(edges_to_sum[:-1]+edges_to_sum[1:])
        num = np.matmul(counts.T,centers)
        den = np.sum(counts,axis=0)
        num_err = np.matmul(errors.T**2,centers**2)**0.5
        den_err = np.sum(errors**2, axis=0)**0.5
        r_val = num/den
        r_err = ((num_err/den)**2 + (den_err*num/den**2.0)**2.0)**0.5
        hnew = Hist1D()
        hnew._counts = r_val
        hnew._errors = r_err
        hnew._edges = edges
        return hnew

    def x_profile(self):
        xedges, yedges = self._edges
        return self._calculate_profile(self._counts, self._errors, yedges, xedges)

    def y_profile(self):
        xedges, yedges = self._edges
        return self._calculate_profile(self._counts.T, self._errors.T, xedges, yedges)

