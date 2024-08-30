#!/usr/bin/env python3
import pyro
import torch
import numpy as np
import muon as mu
from muon import MuData
from sklearn.preprocessing import LabelEncoder
from anndata import AnnData
from ..models.SOFA import SOFA
import pandas as pd
import scanpy as sc
from typing import Union
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import gseapy as gp

def get_ad(data: pd.DataFrame, 
           llh: str="gaussian", 
           select_hvg: bool=False, 
           log: bool=False, 
           scale: bool=False, 
           scaling_factor: Union[None, float]=None
           ) -> AnnData:
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
        whether to select highly variable features.
    log: bool, optional
        whether to log transform the data.
    scale: bool, optional
        whether to center and scale the data.
    scaling_factor: float, optional
        The scaling factor to scale the likelihood for this view. 
    Returns:
    --------
    adata : AnnData
        The converted AnnData object.
    """
    data =data.loc[:,~data.columns.duplicated()]
    data_ = data.loc[~np.all(pd.isnull(data), axis=1),:]
    data = data.loc[:,~np.any(pd.isnull(data_), axis=0)]
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
    
    if scaling_factor is not None:
        adata.uns["scaling_factor"] = scaling_factor
    else:
        adata.uns["scaling_factor"] = 0.1

    adata.obsm["mask"] = adata.obsm["mask"].values
    

    return adata


def calc_var_explained(X_pred, X):
    """
    Calculate the fraction of variance of each view 
    that is explained by each factor.

    Parameters
    ----------
    X_pred : numpy.array
        Predicted X.
    X : numpy.array
        Input X.

    Returns
    -------
    numpy.array
        Array containing the fraction of variance of each view 
        that is explained by each factor.
    """
    vexp = []
    for i in range(len(X_pred)):
        num = np.sum(np.square(X-X_pred[i]))
        denom = np.sum(np.square(X))
        vexp.append(1 - num/denom)
    vexp = np.stack(vexp)
    vexp[vexp < 0] = 0
    return vexp

def calc_var_explained_view(X_pred, X):
    """
    Calculate the total fraction of variance explained of each view.

    Parameters
    ----------
    X_pred : np.array
        Predicted X.
    X : np.array
        Input X.

    Returns
    -------
    numpy.array
        Array containing the total fraction of variance of each view.
    """
    vexp = []
    vexp = 1 - np.sum((np.square(X-X_pred)))/np.sum(np.square(X))
    if vexp < 0:
        vexp = 0
    return vexp

def get_W(model: SOFA,
          view: str
          )-> pd.DataFrame:
    """
    Get the loadings of the model for a specific view.
    
    Parameters
    ----------
    model : SOFA
        The trained SOFA model.
    view : str
        Name of the view to get the loadings for.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loadings of the model for the specified view.
    """
    if hasattr(model, f"W"):
        W = pd.DataFrame(model.W[model.views.index(view)], columns = model.Xmdata.mod[model.views[view]].var_names)
    else:
        model.W = model.predict("W")
        W = pd.DataFrame(model.W[model.views.index(view)], columns = model.Xmdata.mod[model.views[view]].var_names)
    return W

def get_Z(model: SOFA,
          )-> pd.DataFrame:
    """
    Get the loadings of the model for a specific view.
    
    Parameters
    ----------
    model : SOFA
        The trained SOFA model.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the loadings of the model for the specified view.
    """

    col_labels = np.array([f"Factor_{i}" for i in range(model.num_factors)], dtype=object)
    if model.Ymdata is not None:
        guided_factors = list(model.Ymdata.mod.keys())
        for i in range(len(guided_factors)):
            s =  " (" + guided_factors[i] + ")"
            col_labels[model.design.cpu().numpy()[i,:]==1] = col_labels[model.design.cpu().numpy()[i,:]==1] + s

    if hasattr(model, f"Z"):
        Z = pd.DataFrame(model.Z, columns = col_labels)
    else:
        model.Z = model.predict("Z")
        Z = pd.DataFrame(model.Z, columns =  col_labels)
    return Z

def get_top_loadings(model,view, factor=0, sign="+", top_n=100):
    """
    Get the top_n loadings of the model for a specific view.
    
    Parameters
    ----------
    model : SOFA
        The trained SOFA model.
    view : str
        Name of the view to get the loadings for.
    factor : int
        Index of the factor to get the top loadings for.
    sign : str
        Sign of the loadings to get. Default is "+".
    top_n : int
        Number of top loadings to get. Default is 100.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the top_n loadings of the model for the specified view.
    """
    assert(sign=="+" or sign=="-")
    W = get_W(model, view)
    W = W.loc[factor,:]
    if sign == "+":
        idx = np.argpartition(W, -top_n)[-top_n:]
        topW = W.index[idx]
    elif sign=="-":
        idx = np.argpartition(W*-1, -top_n)[-top_n:]
        topW = W.index[idx]
    return topW

def get_gsea_enrichment(gene_list, db):
    """
    Get gene set enrichment analysis results based on a gene_list using gseapy.

    Parameters
    ----------
    gene_list : list
        List of strings containing gene names.
    db : list
        List of strings containing database names to be used for enrichment analysis.

    Returns
    -------
    Enrichr object
        Enrichr object containing the results of the enrichment analysis.
    """
    enr = gp.enrichr(gene_list=gene_list, # or "./tests/data/gene_list.txt",
                     gene_sets=[db],
                     organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                     outdir=None, # don't write to disk
                    )
    return enr

def get_rmse(model):
    """
    Calculate the root mean squared error of X of the model.

    Parameters
    ----------
    model : SOFA
        THe trained SOFA model.

    Returns
    -------
    float
        The root mean squared error of X of the model.
    """
    rmse = 0
    for i,j in zip(model.X, model.X_pred):
        i = i.cpu().numpy()
        rmse += np.sqrt(np.sum(np.square(i-j))/(i.shape[0]*i.shape[1]))
    return rmse

def get_guide_error(model):
    """
    Calculate the root mean squared error for continuous, binary crossentropy for binary or 
    categorical cross entropy for categorical Y of the model.

    Parameters
    ----------
    model : SOFA
        The trained SOFA model.

    Returns
    -------
    float
        The root mean squared error for continuous, binary crossentropy for binary or 
        categorical cross entropy for categorical Y of the model.
    """
    rmse = []
    for ix, (i,j) in enumerate(zip(model.Y, model.Y_pred)):
        if model.target_llh[ix] == "gaussian":
            i = i.cpu().numpy()
            rmse.append(np.sqrt(np.sum(np.square(i-j))/(i.shape[0]*i.shape[1])))
        elif model.target_llh[ix] == "bernoulli":
            i = i.cpu().numpy()
            rmse.append(log_loss(i,sigmoid(j)))
        elif model.target_llh[ix] == "multinomial":
            i = i.cpu().numpy()
            rmse.append(log_loss(i,softmax(j)))
    return rmse


def save_model(model, file_prefix):
    """
    Saves a model as h5mu and save files to disk.
    Model hyperparameters, input data and predictions are saved in the h5mu file 
    and the model parameters are saved in the save file.
    Both files are needed to load a model and continue training.

    Parameters
    ----------
    model : SOFA
        The trained SOFA model.
    file_prefix : str
        Filename prefix to save the model as h5mu and save files.
    Returns
    -------
    tuple(str,str)
        Filenames of the saved h5mu and save files.
    """

    model_mdata = model.save_as_mudata()
    model_mdata.write(file_prefix+".h5mu")
    
    dict_ = pyro.get_param_store()
    dict_.save(file_prefix +".save")
    return file_prefix +".h5mu", file_prefix +".save"
    

    
def load_model(file_prefix):
    """
    Load a saved model from disk. 
    The function requires an h5mu and a save file to load model.

    Parameters
    ----------
    file_prefix : str
        Filename prefix to save the model as h5mu and save files.

    Returns
    -------
    SOFA
        The loaded SOFA model.
    """
    mdata = mu.read(file_prefix +".h5mu")
    if "target_mod" in list(mdata.uns.keys()):
        Ymdata = MuData({i:mdata.mod[i] for i in mdata.uns["target_mod"]})
        Xmdata = MuData({i:mdata.mod[i] for i in mdata.mod if i not in mdata.uns["target_mod"]})
        target_scale = mdata.uns["target_scale"]
        design = mdata.uns["input_design"]
    else:
        Xmdata = MuData({i:mdata.mod[i] for i in mdata.mod})
        Ymdata = None
        design = np.array(0)
        target_scale= None
    num_factors = mdata.uns["input_num_factors"]
    # TODO find way to save and load mixed column metadata
    #metadata = mdata.uns["metadata"]
    horseshoe = mdata.uns["horseshoe"]

    seed = mdata.uns["seed"]
    model = SOFA(Xmdata, 
                  num_factors=num_factors, 
                  Ymdata = Ymdata,
                  design = torch.tensor(design),
                  device=torch.device('cuda'),
                  horseshoe=horseshoe,
                  subsample=0,
                  target_scale=target_scale,
                  seed=seed)
    model.Z = mdata.uns["Z"]
    W = [mdata.uns[f"W_{i}"] for i in Xmdata.mod]
    model.W = W
    model.X_pred =[mdata.uns[f"X_{i}"] for i in range(len(Xmdata.mod))]
    model.history = mdata.uns["history"]
    
    # load pyro paramstore
    dict_ = pyro.get_param_store()
    dict_.load(file_prefix+".save")
    return model
