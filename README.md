<a href="https://doi.org/10.5281/zenodo.3973431"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3973431.svg" alt="DOI"></a>

# Weighted Ensemble Python: wepy

![Wepy Logo](./info/logo/wepy.svg)


[Documentation](https://adicksonlab.github.io/wepy/index.html)

Modular implementation and framework for running weighted ensemble (WE)
simulations in pure python, where the aim is to have simple things
simple and complicated things possible. The latter being the priority.

The goal of the architecture is that it should be highly modular to
allow extension, but provide a "killer app" for most uses that just
works, no questions asked.

Comes equipped with support for
[OpenMM](https://github.com/pandegroup/openmm) molecular dynamics,
parallelization using multiprocessing, the
[WExplore](http://pubs.acs.org/doi/abs/10.1021/jp411479c) and
[REVO](https://pubmed.ncbi.nlm.nih.gov/31255090/) (Resampling
Ensembles by Variance Optimization) resampling algorithms, and an HDF5
file format and library for storing and querying your WE datasets that
can be used from the command line.

The deeper architecture of `wepy` is intended to be loosely coupled,
so that unforeseen use cases can be accomodated, but tightly
integrated for the most common of use cases, i.e. molecular dynamics.

This allows freedom for fast development of new methods.

## Installation

Also see: [Installation Instructions](info/installation.org)

```shell
pip install wepy

# for openmm and MD related packages
pip install 'wepy[md]'
```

## Citations

Current [Zenodo DOI](https://zenodo.org/badge/latestdoi/101077926).

Cite software as:

```
Samuel D. Lotz, Nazanin Donyapour, Alex Dickson, Tom Dixon, Nicole Roussey, & Rob Hall. (2020, August 4). ADicksonLab/wepy: 1.0.0 Major version release (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3973431
```

Accompanying journal article:

- [ACS Omega](https://pubs.acs.org/doi/abs/10.1021/acsomega.0c03892) article






