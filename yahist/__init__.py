from .hist1d import Hist1D
from .hist2d import Hist2D
from .utils import set_default_style, register_with_dask

register_with_dask([Hist1D, Hist2D])
