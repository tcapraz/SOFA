# Semi-supervised Omics Factor Analysis (SOFA)

# Introduction

Here we present semi-supervised probabilistic Factor Analysis (SOFA), a multi-omics integration method, which infers a set of low dimensional latent factors that represent the main sources of variability. SOFA enables the discovery of primary sources of variation while adjusting for known covariates and simultaneously disentangling variation that is shared between multiple omics modalities and specific to single modalities. The SOFA method is implemented in python using the Pyro framework for probabilistic programming.


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


