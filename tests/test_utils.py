import pytest
from sofa.utils.utils import get_ad, calc_var_explained, calc_var_explained_view, get_W, get_Z, get_top_loadings, get_gsea_enrichment, get_rmse, get_guide_error, save_model, load_model
import pandas as pd
import numpy as np
from anndata import AnnData
from sofa.models.SOFA import SOFA
import gseapy as gp
import torch

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
    assert isinstance(vexp, np.ndarray)
    assert vexp.shape == (2,)
    assert np.all(vexp == 1)

    # Test case 2: X_pred and X with different shape
    X_pred = np.array([[1, 2, 3], [4, 5, 6]])
    X = np.array([[1, 2, 3]])
    vexp = calc_var_explained(X_pred, X)
    assert isinstance(vexp, np.ndarray)
    assert vexp.shape == (2,)
    assert np.all(vexp == 1)

    # Add more test cases...

# Test calc_var_explained_view function
def test_calc_var_explained_view():
    # Test case 1: X_pred and X with same shape
    X_pred = np.array([1, 2, 3])
    X = np.array([1, 2, 3])
    vexp = calc_var_explained_view(X_pred, X)
    assert isinstance(vexp, float)
    assert vexp == 1

    # Test case 2: X_pred and X with different shape
    X_pred = np.array([1, 2, 3])
    X = np.array([1, 2])
    vexp = calc_var_explained_view(X_pred, X)
    assert isinstance(vexp, float)
    assert vexp == 1

    # Add more test cases...

# Test get_W function
def test_get_W():
    # Test case 1: model with existing W attribute
    model = SOFA()
    model.W = [pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})]
    W = get_W(model, 'view1')
    assert isinstance(W, pd.DataFrame)
    assert W.shape == (1, 2)
    assert W.columns.tolist() == ['A', 'B']

    # Test case 2: model without existing W attribute
    model = SOFA()
    model.predict = lambda x: [pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})]
    W = get_W(model, 'view1')
    assert isinstance(W, pd.DataFrame)
    assert W.shape == (1, 2)
    assert W.columns.tolist() == ['A', 'B']

    # Add more test cases...

# Test get_Z function
def test_get_Z():
    # Test case 1: model with existing Z attribute
    model = SOFA()
    model.Z = np.array([[1, 2, 3]])
    Z = get_Z(model)
    assert isinstance(Z, pd.DataFrame)
    assert Z.shape == (1, 3)

    # Test case 2: model without existing Z attribute
    model = SOFA()
    model.predict = lambda x: np.array([[1, 2, 3]])
    Z = get_Z(model)
    assert isinstance(Z, pd.DataFrame)
    assert Z.shape == (1, 3)

    # Add more test cases...

# Test get_top_loadings function
def test_get_top_loadings():
    # Test case 1: model with positive loadings
    model = SOFA()
    model.W = [pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})]
    topW = get_top_loadings(model, 'view1', factor=0, sign='+', top_n=1)
    assert isinstance(topW, pd.Index)
    assert topW.tolist() == ['B']

    # Test case 2: model with negative loadings
    model = SOFA()
    model.W = [pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})]
    topW = get_top_loadings(model, 'view1', factor=0, sign='-', top_n=1)
    assert isinstance(topW, pd.Index)
    assert topW.tolist() == ['A']

    # Add more test cases...

# Test get_gsea_enrichment function
def test_get_gsea_enrichment():
    # Test case 1: gene_list, db, and background as lists
    gene_list = ['gene1', 'gene2', 'gene3']
    db = ['db1', 'db2']
    background = ['gene1', 'gene2', 'gene3', 'gene4']
    enr = get_gsea_enrichment(gene_list, db, background)
    assert isinstance(enr, gp.enrichr.Enrichr)
    # Add assertions for the returned Enrichr object

    # Test case 2: gene_list, db, and background as file paths
    gene_list = './tests/data/gene_list.txt'
    db = './tests/data/db.txt'
    background = './tests/data/background.txt'
    enr = get_gsea_enrichment(gene_list, db, background)
    assert isinstance(enr, gp.enrichr.Enrichr)
    # Add assertions for the returned Enrichr object

    # Add more test cases...

# Test get_rmse function
def test_get_rmse():
    # Test case 1: model with X and X_pred of same shape
    model = SOFA()
    model.X = [torch.tensor([[1, 2, 3], [4, 5, 6]])]
    model.X_pred = [torch.tensor([[1, 2, 3], [4, 5, 6]])]
    rmse = get_rmse(model)
    assert isinstance(rmse, float)
    assert rmse == 0

    # Test case 2: model with X and X_pred of different shape
    model = SOFA()
    model.X = [torch.tensor([[1, 2, 3], [4, 5, 6]])]
    model.X_pred = [torch.tensor([[1, 2, 3]])]
    rmse = get_rmse(model)
    assert isinstance(rmse, float)
    assert rmse == 0

    # Add more test cases...

# Test get_guide_error function
def test_get_guide_error():
    # Test case 1: model with continuous target_llh
    model = SOFA()
    model.Y = [torch.tensor([[1, 2, 3], [4, 5, 6]])]
    model.Y_pred = [torch.tensor([[1, 2, 3], [4, 5, 6]])]
    model.target_llh = ['gaussian']
    error = get_guide_error(model)
    assert isinstance(error, list)
    assert len(error) == 1
    assert error[0] == 0

    # Test case 2: model with binary target_llh
    model = SOFA()
    model.Y = [torch.tensor([[1, 0, 1], [0, 1, 0]])]
    model.Y_pred = [torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.8, 0.3]])]
    model.target_llh = ['bernoulli']
    error = get_guide_error(model)
    assert isinstance(error, list)
    assert len(error) == 1
    # Add assertions for the calculated error

    # Add more test cases...

# Test save_model function
def test_save_model():
    # Test case 1: model with default parameters
    model = SOFA()
    file_prefix = 'model'
    h5mu_file, save_file = save_model(model, file_prefix)
    assert isinstance(h5mu_file, str)
    assert isinstance(save_file, str)
    # Add assertions for the existence of the saved files

    # Add more test cases...

# Test load_model function
def test_load_model():
    # Test case 1: model with default parameters
    file_prefix = 'model'
    model = load_model(file_prefix)
    assert isinstance(model, SOFA)
    # Add assertions for the loaded model

    # Add more test cases...

# Add more test cases...
