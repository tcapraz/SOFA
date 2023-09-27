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
from sklearn import preprocessing
import anndata as ad
from itertools import product
from spFA.spFA.spFA import spFA
from spFA.spFA.plots import calc_var_explained
import scipy.stats as stats
import pandas as pd

def get_ad(var, name, llh="gaussian"):
    if llh == "multinomial":
        label_map= {lab:i for i,lab in enumerate(np.unique(var))}
        var_ = np.array([label_map[i] for i in var])
        adata = ad.AnnData(var_.reshape(-1,1)) 
    elif llh == "bernoulli":
        if type(var) !=  int:
            assert len(np.unique(var)) == 2
            label_map= {lab:i for i,lab in enumerate(np.unique(var))}
            var_ = np.array([float(label_map[i]) for i in var])
            adata = ad.AnnData(var_.reshape(-1,1))
    else:
        adata = ad.AnnData(var.values.reshape(-1,1))
    adata.obs_names = var.index.astype(str).tolist()
    adata.obsm["mask"] = ~np.isnan(adata.X.flatten())  
    adata.var_names = [name]
    adata.uns["llh"] = llh
    if llh == "multinomial" or llh == "bernoulli":
        adata.uns["label_map"] = label_map
    return adata

def tune_scaling(Xmdata, Ymdata, design,n_factors,metadata, scale_range, metavar, n_steps=3500):
    vexp_out = {}
    cor_out = {}
    f1_out = {}

    targets = Ymdata.mod
    scale = []
    for i in product(*list(scale_range.values())):
        scale.append(i)
    for s in scale:
        model = spFA(Xmdata=Xmdata,
                 Ymdata = Ymdata,
                 design=design,
                 num_factors=n_factors,
                 device=torch.device('cuda'),
                 ard = False,
                 horseshoe = True,
                 subsample=0, 
                 metadata=metadata,
                 target_scale=s)
        model.fit_spFA(n_steps=n_steps, lr=0.01)
        X = [i.cpu().numpy() for i in model.X]
        vexp = []
        for i in range(len(X)):
            mask = model.Xmask[i].cpu().numpy()
            X_pred_factor = []
            for j in range(len(Ymdata.mod)):
                X_pred_factor.append(model.Z[mask,j, np.newaxis] @ model.W[i][np.newaxis,j,:])
            vexp.append(calc_var_explained(X_pred_factor, X[i][mask,:]).reshape(len(Ymdata.mod),1))
        vexp = np.hstack(vexp)
        vexp = np.sum(vexp, axis=1)
        
        cor = []
        for ix, i in enumerate(metavar):
            var = model.metadata[i].values
            mask = ~model.metadata[i].isna()
            if var.dtype.type is np.string_ or var.dtype.type is np.object_ or var.dtype.type is pd.core.dtypes.dtypes.CategoricalDtypeType:
                lmap = {l:j for j,l in enumerate(np.unique(var[mask]))}
                y = np.array([lmap[l] for l in var[mask]])
            else:
                y = var[mask]
            cor.append(stats.pearsonr(model.Z[mask,ix], y)[0])
        f1 = []
        for i in range(len(Ymdata.mod)):
            f1.append(len(Ymdata.mod)/(1/vexp[i] + 1/abs(cor[i])))
        vexp_out[s] = vexp.tolist()
        cor_out[s] = cor
        f1_out[s] = f1

    return vexp_out, cor_out, f1_out