# Semi-supervised Omics Factor Analysis (SOFA)


[![PyPI version](https://badge.fury.io/py/biosofa.svg)](https://badge.fury.io/py/biosofa)

# Introduction

Here we present Semi-supervised Omics Factor Analysis (SOFA), a multi-omics integration method, that incorporates known sources of variation into the model and focuses the latent factor discovery on novel sources of variation. The SOFA method is implemented in Python using the Pyro framework for probabilistic programming.

![The SOFA model](https://github.com/tcapraz/SOFA/blob/main/docs/model_schematic.png?raw=true)

**We are still working on improvements to the SOFA package. Please expect breaking changes. If you find a bug or have ideas how to make the user experience of SOFA smoother please open an issue.**

# Installation

To install `SOFA` first create `Python 3.8` environment e.g. by

```
conda create --name sofa-env python=3.8
conda activate sofa-env
```

and install the package using 

```
pip install biosofa
```



# How to use `SOFA` for multi-omics analyses

A detailed manual with examples and how to use `SOFA` can be found here https://tcapraz.github.io/SOFA/index.html.


# How to cite `SOFA`

> **Semi-supervised Omics Factor Analysis (SOFA) disentangles known sources of variation from latent factors in multi-omics data**
>
> Capraz, T., Vöhringer, H.S. and Huber, W.
>
> *bioRxiv* 2024. doi: [10.1101/2024.10.10.617527](https://doi.org/10.1101/2024.10.10.617527).
