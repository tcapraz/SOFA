import pytest
from sofa.utils.utils import get_ad, calc_var_explained,  get_loadings, get_factors, get_top_loadings, get_gsea_enrichment, get_rmse, get_guide_error, save_model, load_model
import pandas as pd
import numpy as np
from anndata import AnnData
from sofa.models.SOFA import SOFA
import gseapy as gp
import torch
from muon import MuData
import pyro

# Test get_ad function
def test_get_ad():
    # Test case 1: data with default parameters
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    adata = get_ad(data)
    assert isinstance(adata, AnnData)
    assert adata.shape == (3, 2)
    assert adata.var_names.tolist() == ['A', 'B']
    assert adata.obs_names.tolist() == ['0', '1', '2']
    assert 'mask' in adata.obsm.keys()
    assert adata.uns['llh'] == 'gaussian'
    assert adata.uns['scaling_factor'] == 0.1

    # Test case 2: data with log transformation
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    adata = get_ad(data, log=True)
    assert isinstance(adata, AnnData)
    assert adata.shape == (3, 2)
    assert adata.var_names.tolist() == ['A', 'B']
    assert adata.obs_names.tolist() == ['0', '1', '2']
    assert 'mask' in adata.obsm.keys()
    assert adata.uns['llh'] == 'gaussian'
    assert adata.uns['scaling_factor'] == 0.1
    assert np.all(adata.X >= 0)

    # Add more test cases...

# Test calc_var_explained function
def test_calc_var_explained():
    # Test case 1: X_pred and X with same shape
    X_pred = np.array([[1, 2, 3], [4, 5, 6]])
    X = np.array([[1, 2, 3], [4, 5, 6]])
    vexp = calc_var_explained(X_pred, X)
    assert isinstance(vexp, float)
    assert np.all(vexp == pytest.approx(1))






    # Add more test cases...

# Test get_loadings function
def test_get_loadings(sample_model, sample_data):
    # Test case 1: model with existing W attribute
    model = sample_model
    W = get_loadings(model, 'view1')
    assert isinstance(W, pd.DataFrame)
    assert W.shape == (model.num_factors, model.X[0].shape[1])

    # Test case 2: model without existing W attribute
    Xmdata = MuData({"view1": sample_data, "view2": sample_data})
    num_factors = 10
    Ymdata = None
    design = None
    device = "cpu"
    horseshoe = True
    update_freq = 200
    subsample = 0
    metadata = None
    verbose = True
    horseshoe_scale_feature = 1
    horseshoe_scale_factor = 1
    horseshoe_scale_global = 1
    seed = None

    # check if prediction works
    model = SOFA(Xmdata, num_factors, Ymdata, design, device, horseshoe, update_freq, subsample, metadata, verbose, horseshoe_scale_feature, horseshoe_scale_factor, horseshoe_scale_global, seed)

    W = get_loadings(model, 'view1')
    assert isinstance(W, pd.DataFrame)
    assert W.shape == (num_factors, sample_data.X.shape[1])
    assert W.columns.tolist() == sample_data.var_names.tolist()
    pyro.clear_param_store()



# Test get_factors function
def test_get_factors(sample_model, sample_data):

    Z = get_factors(sample_model)
    assert isinstance(Z, pd.DataFrame)
    assert Z.shape == (sample_model.num_samples, sample_model.num_factors)

    # test if prediction works
    # Test case 2: model without existing W attribute
    Xmdata = MuData({"view1": sample_data, "view2": sample_data})
    num_factors = 10
    Ymdata = None
    design = None
    device = "cpu"
    horseshoe = True
    update_freq = 200
    subsample = 0
    metadata = None
    verbose = True
    horseshoe_scale_feature = 1
    horseshoe_scale_factor = 1
    horseshoe_scale_global = 1
    seed = None

    # check if prediction works
    model = SOFA(Xmdata, num_factors, Ymdata, design, device, horseshoe, update_freq, subsample, metadata, verbose, horseshoe_scale_feature, horseshoe_scale_factor, horseshoe_scale_global, seed)

    Z = get_factors(model)

    assert isinstance(Z, pd.DataFrame)
    assert Z.shape == (model.num_samples, model.num_factors)
    pyro.clear_param_store()

    # Add more test cases...

# Test get_top_loadings function
def test_get_top_loadings(sample_model):
    # Test case 1: model with positive loadings
    model = sample_model
    topW = get_top_loadings(model, 'view1', factor=1, sign='+', top_n=1)
    assert isinstance(topW, list)
    pyro.clear_param_store()





# Test get_gsea_enrichment function
#def test_get_gsea_enrichment():
    # Test case 1: gene_list, db, and background as lists
#    gene_list = ["IGF2" 	,"DLK1" ,	"CYP17A1" ]
#    background = ["IGF2" 	,"DLK1" ,	"CYP17A1" ,	"APOE" ,	"SLPI" 	,"CYP11B1" ,	"STAR"]
#    db = "GO_Biological_Process_2023"

#    enr = get_gsea_enrichment(gene_list, db, background)
#    assert isinstance(enr, gp.Enrichr)

    

# Test get_rmse function
def test_get_rmse(sample_model):
    rmse = get_rmse(sample_model)
    assert isinstance(rmse, dict)
    assert isinstance(rmse['view1'], float)
    pyro.clear_param_store()



# Test get_guide_error function
def test_get_guide_error(sample_model, sample_data):
    # Xmdata = MuData({"view1": sample_data, "view2": sample_data})
    # num_factors = 10
    # Ymdata = None
    # design = None
    # device = "cpu"
    # horseshoe = True
    # update_freq = 200
    # subsample = 0
    # metadata = None
    # verbose = True
    # horseshoe_scale_feature = 1
    # horseshoe_scale_factor = 1
    # horseshoe_scale_global = 1
    # seed = None

    # # check if prediction works
    # model = SOFA(Xmdata, num_factors, Ymdata, design, device, horseshoe, update_freq, subsample, metadata, verbose, horseshoe_scale_feature, horseshoe_scale_factor, horseshoe_scale_global, seed)
    # print(sample_model.num_factors)
    # print(sample_model.num_features)

    error = get_guide_error(sample_model)
    assert isinstance(error, dict)
    assert isinstance(error["guide1"], float) 
    sample_model.Ymdata = None
    with pytest.raises(Exception) as e_info:
        error = get_guide_error(sample_model)


# Test save_model function
def test_save_model(sample_model):
    # Test case 1: model with default parameters
    model = sample_model
    file_prefix = 'model'
    h5mu_file, save_file = save_model(model, file_prefix)
    assert isinstance(h5mu_file, str)
    assert isinstance(save_file, str)
    pyro.clear_param_store()

    # Add assertions for the existence of the saved files

    # Add more test cases...

# Test load_model function
def test_load_model(sample_model):
    # Test case 1: model with default parameters
    file_prefix = 'model'
    model = sample_model
    h5mu_file, save_file = save_model(model, file_prefix)
    model = load_model(file_prefix)
    assert isinstance(model, SOFA)

    # Add assertions for the loaded model

