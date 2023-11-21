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
import scanpy as sc
from typing import Union
import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder, StandardScaler
import gseapy as gp
import pickle

def get_ad(data: pd.DataFrame, llh: str="gaussian", select_hvg: bool=False, log: bool=False, scale: bool=False) -> AnnData:
    """
    Convert a numpy array to an AnnData object.

    Parameters:
    -----------
    data : pandas DataFrame
        The input data to be converted to AnnData object.
    name : str
        The name of the variable.
    llh : str, optional
        The likelihood of the data. Default is "gaussian".
    select_hvg: bool, optional
        whether to select highly variable features

    Returns:
    --------
    adata : AnnData
        The converted AnnData object.
    """
    data =data.loc[:,~data.columns.duplicated()]
    mask = ~np.any(pd.isnull(data), axis=1)
    mask.index = mask.index.astype(str)
    if llh == "multinomial" or llh == "bernoulli":
        if type(data) !=  int:
            label_encoder = LabelEncoder() 
            # Apply LabelEncoder to each column
            encoded_data = data.values.flatten()
            encoded_data = label_encoder.fit_transform(encoded_data)
            encoded_data = encoded_data.reshape(data.shape)
            adata = AnnData(encoded_data, dtype=np.float32)
            label_mapping = {label: encoded_label for label, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    else:
        encoded_data = data.values

    if data.shape[1] == 1:
        adata = AnnData(encoded_data.reshape(-1,1), dtype=np.float32)
        adata.var_names = data.columns
    else:
        adata = AnnData(encoded_data, dtype=np.float32) 
        adata.var_names = data.columns
    data.index = data.index.astype(str)
    adata.obs_names = data.index.tolist()
    if log:
        sc.pp.log1p(adata)
    adata.obsm["mask"] = mask

    if select_hvg:
        adata_filtered = adata[~np.all(data.isna(),axis=1),:]

        adata = adata[:,~np.any(np.isnan(adata_filtered.X),axis=0)]

        adata_filtered = adata_filtered[:,~np.any(np.isnan(adata_filtered.X),axis=0)]

        sc.pp.highly_variable_genes(adata_filtered, n_top_genes=2000)
        adata = adata[:,adata_filtered.var["highly_variable"]]
        adata.var["highly_variable"] = adata_filtered.var["highly_variable"]
    if scale: 
        scaler = StandardScaler()
        adata.X = scaler.fit_transform(adata.X)

    adata.X[adata.obsm["mask"] == False] = 0
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

def get_top_loadings(model,factor=0, view=0, sign="+", top_n=100):
    
    assert(sign=="+" or sign=="-")
    W = pd.DataFrame(model.W[view], columns = model.Xmdata.mod[list(model.Xmdata.mod.keys())[view]].var_names)
    W = W.loc[factor,:]

    if sign == "+":
        idx = np.argpartition(W, -top_n)[-top_n:]

        topW = W.index[idx]


    elif sign=="-":
        idx = np.argpartition(W*-1, -top_n)[-top_n:]

        topW = W.index[idx]
    return topW

def get_gsea_enrichment(gene_list, db):
    # if you are only intrested in dataframe that enrichr returned, please set outdir=None
    enr = gp.enrichr(gene_list=gene_list, # or "./tests/data/gene_list.txt",
                     gene_sets=[db],
                     organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                     outdir=None, # don't write to disk
                    )
    return enr

def get_rmse(model):
    rmse = 0
    for i,j in zip(model.X, model.X_pred):
        i = i.cpu().numpy()
        rmse += np.sqrt(np.sum(np.square(i-j))/(i.shape[0]*i.shape[1]))
    return rmse

def get_rmse_target(model):
    rmse = 0
    for i,j in zip(model.Y, model.Y_pred):
        i = i.cpu().numpy()
        rmse += np.sqrt(np.sum(np.square(i-j))/(i.shape[0]*i.shape[1]))
    return rmse

def save(model, model_filename, param_filename):
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    dict_ = pyro.get_param_store()

    dict_.save(param_filename)

def load(model_filename, param_filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    dict_ = pyro.get_param_store()

    dict_.load(param_filename)
    return model
    