## Yet another histogram

![Python application](https://github.com/aminnj/yahist/workflows/Python%20application/badge.svg)

### Overview

A histogram object with simple manipulations, plotting, and fitting.

```python
h = (Hist1D(np.random.normal(0, 1, 1000), bins=100, label="data")
     .rebin(2)
     .normalize()
    )
h.plot(show_errors=True, color="k")
h.fit("peak * np.exp(-(x-mu)**2 / (2*sigma**2))")
```

![](examples/plot1.png)

Much more functionality is showcased in the example notebook below.

### Examples

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aminnj/yahist/master?filepath=examples%2Fbasic.ipynb)

(static [nbviewer](https://nbviewer.jupyter.org/url/github.com/aminnj/yahist/blob/master/examples/basic.ipynb) if Binder is slow)

### Installation

```bash
pip install yahist
```
or to install the latest version directly from github
```bash
pip install git+git://github.com/aminnj/yahist.git#egg=yahist -U
```
