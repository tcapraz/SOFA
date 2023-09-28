#!/usr/bin/env python3
import pyro
import torch
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive
import numpy as np
from pyro.optim import Adam
import torch.nn as nn
from tqdm import tqdm
import muon as mu
from muon import MuData
from sklearn.preprocessing import LabelEncoder
import anndata as ad
from itertools import product
from spFA.spFA.spFA import spFA
from spFA.spFA.plots import calc_var_explained
import scipy.stats as stats
import pandas as pd

def get_ad(data, name, llh="gaussian"):
    if llh == "multinomial":
        label_encoders = [LabelEncoder() for _ in range(data.shape[1])]
        # Apply LabelEncoder to each column
        encoded_data = np.zeros_like(data, dtype=int)
        for i in range(data.shape[1]):
            encoded_data[:, i] = label_encoders[i].fit_transform(data[:, i])

    elif llh == "bernoulli":
        if type(var) !=  int:
            label_encoders = [LabelEncoder() for _ in range(data.shape[1])]
            # Apply LabelEncoder to each column
            encoded_data = np.zeros_like(data, dtype=int)

            for i in range(data.shape[1]):
                encoded_data[:, i] = label_encoders[i].fit_transform(data[:, i])

            adata = ad.AnnData(encoded_data)
    else:
        encoded_data = data

    if data.shape[1] == 1:
        adata = ad.AnnData(encoded_data.values.reshape(-1,1))
    else:
        adata = ad.AnnData(encoded_data) 

    adata.obs_names = data.index.astype(str).tolist()
    adata.obsm["mask"] = ~np.any(np.isnan(data.values), axis=1)
    adata.var_names = [name]
    adata.uns["llh"] = llh
    if llh == "multinomial" or llh == "bernoulli":
        adata.uns["label_map"] = label_map
    return adata

def make_adata(data, llh="gaussian"):
    adata = ad.AnnData(data)
    adata.obs_names = data.index.astype(str).tolist()
    adata.var_names = data.columns
    adata.obsm["mask"] = ~np.any(np.isnan(data.values), axis=1)
    adata.uns["llh"] = llh
    return adata

