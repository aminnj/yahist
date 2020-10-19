#!/usr/bin/env sh

pip3 install --user sphinx sphinxcontrib-napoleon pydata-sphinx-theme

sphinx-apidoc yahist/ -f -o tempdocs/ -F --extension sphinx.ext.napoleon
(cd tempdocs && sphinx-build . ../docs -D html_theme=nature)
# or -D html_theme=pydata_sphinx_theme
