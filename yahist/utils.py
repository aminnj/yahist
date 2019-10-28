from __future__ import print_function

import matplotlib
import numpy as np

def is_listlike(obj):
    return (hasattr(obj, "__array__") or type(obj) in [list, tuple])

def set_default_style():
    from matplotlib import rcParams
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "Bitstream Vera Sans", "DejaVu Sans"]
    rcParams['legend.fontsize'] = 11
    rcParams['legend.labelspacing'] = 0.2
    rcParams['hatch.linewidth'] = 0.5  # https://stackoverflow.com/questions/29549530/how-to-change-the-linewidth-of-hatch-in-matplotlib
    rcParams['axes.xmargin'] = 0.0 # rootlike, no extra padding within x axis
    rcParams['axes.labelsize'] = 'x-large'
    rcParams['axes.formatter.use_mathtext'] = True
    rcParams['legend.framealpha'] = 0.65
    rcParams['axes.labelsize'] = 'x-large'
    rcParams['axes.titlesize'] = 'large'
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['figure.subplot.hspace'] = 0.1
    rcParams['figure.subplot.wspace'] = 0.1
    rcParams['figure.subplot.right'] = 0.96
    rcParams['figure.max_open_warning'] = 0
    rcParams['figure.dpi'] = 100
    rcParams["axes.formatter.limits"] = [-5,4] # scientific notation if log(y) outside this


def compute_darkness(r,g,b,a=1.0):
    # darkness = 1 - luminance
    return a*(1.0 - (0.299*r + 0.587*g + 0.114*b))

def clopper_pearson_error(passed, total, level=0.6827):
    """
    matching TEfficiency::ClopperPearson()
    """
    import scipy.stats
    alpha = 0.5*(1.-level)
    low = scipy.stats.beta.ppf(alpha, passed, total-passed+1)
    high = scipy.stats.beta.ppf(1 - alpha, passed+1, total-passed)
    return low, high

def poisson_errors(obs,alpha=1-0.6827):
    """
    Return poisson low and high values for a series of data observations
    """
    from scipy.stats import gamma
    lows = np.nan_to_num(gamma.ppf(alpha/2,np.array(obs)))
    highs = np.nan_to_num(gamma.ppf(1.-alpha/2,np.array(obs)+1))
    return lows, highs

def binomial_obs_z(data,bkg,bkgerr,gaussian_fallback=True):
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
    nonzeros = (data>1.e-6)
    tau = 1./bkg[nonzeros]/(bkgerr[nonzeros]/bkg[nonzeros])**2.
    auxinf = bkg[nonzeros]*tau
    v = betainc(data[nonzeros],auxinf+1,1./(1.+tau))
    z[nonzeros] = st.norm.ppf(1-v)
    if (data<1.e-6).sum():
        zeros = (data<1.e-6)
        z[zeros] = -(bkg[zeros])/np.hypot(poisson_errors(data[zeros])[1],bkgerr[zeros])
    return z

def nan_to_num(f):
    def g(*args, **kw):
        return np.nan_to_num(f(*args, **kw))
    return g

def ignore_division_errors(f):
    def g(*args, **kw):
        with np.errstate(divide="ignore",invalid="ignore"):
            return f(*args, **kw)
    return g

class TextPatchHandler(object):
    def __init__(self, label_map={}):
        self.label_map = label_map
        super(TextPatchHandler, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        label = orig_handle.get_label()
        fc = orig_handle.get_facecolor()
        ec = orig_handle.get_edgecolor()
        lw = orig_handle.get_linewidth()
        color = "w" if (compute_darkness(*fc) > 0.45) else "k"
        text = self.label_map.get(label,"")
        patch1 = matplotlib.patches.Rectangle([x0, y0], width, height, facecolor=fc, edgecolor=ec, linewidth=lw, transform=handlebox.get_transform())
        patch2 = matplotlib.text.Text(x0+0.5*width,y0+0.45*height,text,transform=handlebox.get_transform(),fontsize=0.55*fontsize, color=color, ha="center",va="center")
        handlebox.add_artist(patch1)
        handlebox.add_artist(patch2)
        return patch1

def register_root_palettes():
    # RGB stops taken from
    # https://github.com/root-project/root/blob/9acb02a9524b2d9d5edb57c519aea4f4ab8022ac/core/base/src/TColor.cxx#L2523

    palettes = {
            "kBird": {
                "reds": [ 1., 0.2082, 0.0592, 0.0780, 0.0232, 0.1802, 0.5301, 0.8186, 0.9956, 0.9764 ],
                "greens": [ 1., 0.1664, 0.3599, 0.5041, 0.6419, 0.7178, 0.7492, 0.7328, 0.7862, 0.9832 ],
                "blues": [ 1., 0.5293, 0.8684, 0.8385, 0.7914, 0.6425, 0.4662, 0.3499, 0.1968, 0.0539 ],
                "stops": np.concatenate([[0.],np.linspace(1.e-6,1.,9)]),
                },
            "kRainbow": {
                "reds": [ 1., 0./255., 5./255., 15./255., 35./255., 102./255., 196./255., 208./255., 199./255., 110./255.],
                "greens": [ 1., 0./255., 48./255., 124./255., 192./255., 206./255., 226./255., 97./255., 16./255., 0./255.],
                "blues": [ 1., 99./255., 142./255., 198./255., 201./255., 90./255., 22./255., 13./255., 8./255., 2./255.],
                "stops": np.concatenate([[0.],np.linspace(1.e-6,1.,9)]),
                },
            "SUSY": {
                "reds": [1.00, 0.50, 0.50, 1.00, 1.00, 1.00],
                "greens": [1.00, 0.50, 1.00, 1.00, 0.60, 0.50],
                "blues": [1.00, 1.00, 1.00, 0.50, 0.40, 0.50],
                "stops": [0.0, 1.e-6, 0.34, 0.61, 0.84, 1.00],
                },
            }

    for key in palettes:
        stops = palettes[key]["stops"]
        reds = palettes[key]["reds"]
        greens = palettes[key]["greens"]
        blues = palettes[key]["blues"]
        cdict = {
            "red": zip(stops,reds,reds),
            "green": zip(stops,greens,greens),
            "blue": zip(stops,blues,blues)
        }
        matplotlib.pyplot.register_cmap(name=key, data=cdict)

