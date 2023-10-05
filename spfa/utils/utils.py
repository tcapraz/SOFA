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
from anndata import AnnData
from itertools import product

import scipy.stats as stats
import pandas as pd

from typing import Union
import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder

def get_ad(
        data: np.ndarray, 
        llh: str="gaussian"
        ) -> AnnData:
    """
    Convert a numpy array to an AnnData object.

    Parameters:
    -----------
    data : np.ndarray
        The input data to be converted to AnnData object.
    name : str
        The name of the variable.
    llh : str, optional
        The likelihood of the data. Default is "gaussian".

    Returns:
    --------
    adata : AnnData
        The converted AnnData object.
    """
    if llh == "multinomial" or llh == "bernoulli":
        if type(data) !=  int:
            label_encoder = LabelEncoder() 
            # Apply LabelEncoder to each column
            encoded_data = data.values.flatten()
            encoded_data = label_encoder.fit_transform(encoded_data)
            encoded_data = encoded_data.reshape(data.shape)
            adata = AnnData(encoded_data)
            label_mapping = {label: encoded_label for label, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    else:
        encoded_data = data.values

    if data.shape[1] == 1:
        adata = AnnData(encoded_data.reshape(-1,1))
        adata.var_names = data.columns
    else:
        adata = AnnData(encoded_data) 
        adata.var_names = data.columns
    data.index = data.index.astype(str)
    adata.obs_names = data.index.tolist()
    adata.obsm["mask"] = ~np.any(pd.isnull(data), axis=1)
    adata.uns["llh"] = llh
    if llh == "multinomial" or llh == "bernoulli":
        adata.uns["label_map"] = label_mapping
    return adata


def calc_var_explained(X_pred, X):
    vexp = []

    for i in range(len(X_pred)):
        num = np.sum(np.square(X-X_pred[i]))
        denom = np.sum(np.square(X))
        vexp.append(1 - num/denom)
    vexp = np.stack(vexp)
    vexp[vexp < 0] = 0
    return vexp

def calc_var_explained_view(X_pred, X):
    vexp = []
    

    vexp = 1 - np.sum((np.square(X-X_pred)))/np.sum(np.square(X))
    if vexp < 0:
        vexp = 0
    return vexp