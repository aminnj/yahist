from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = "1.0.0"

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [
    x.strip().replace("git+", "") for x in all_reqs if x.startswith("git+")
]

setup(
    name="yahist",
    version=__version__,
    description="Yet another histogram object with numpy and matplotlib",
    long_description=long_description,
    url="https://github.com/aminnj/yahist",
    download_url="https://github.com/aminnj/yahist/tarball/" + __version__,
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
    ],
    keywords="",
    packages=find_packages(exclude=["docs", "tests*", "examples"]),
    include_package_data=True,
    author="Nick Amin",
    install_requires=install_requires,
    # extras_require={
    #     "autograd": [],
    #     },
    dependency_links=dependency_links,
    author_email="amin.nj@gmail.com",
)
