# spFA

# Introduction

Here we present semi-supervised probabilistic Factor Analysis (spFA), a multi-omics integration method, which infers a set of low dimensional latent factors that represent the main sources of variability. spFA enables the discovery of primary sources of variation while adjusting for known covariates and simultaneously disentangling variation that is shared between multiple omics modalities and specific to single modalities. The spFA method is implemented in python using the pyro framework for probabilistic programming.


# Installation

To install `spfa` first create `Python 3.8` environment e.g. by

```
conda create --name spfa-env python=3.8
conda activate spfa-env
```

and install the package using 

```
pip install spfa
```



# How to use `spfa` for multi-omics analyses

A detailed manual with examples and how to use `spfa` can be found here https://tcapraz.github.io/spFA/index.html.


