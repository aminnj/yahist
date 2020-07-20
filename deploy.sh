#!/usr/bin/env bash

# edit version in setup.py
rm -f dist/*
python setup.py bdist_wheel
python -m twine upload dist/*
# note, can do `pip install -e .` in this folder to install a "development" version of the package
