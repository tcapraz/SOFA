import pytest
from sofa.plots.plots import plot_loadings, plot_top_loadings, plot_variance_explained, plot_variance_explained_factor, plot_variance_explained_view, plot_factor_covariate_cor, plot_fit, plot_enrichment



def test_plot_loadings(sample_model):
    # Test plot_loadings function
    ax = plot_loadings(sample_model, "view1", 0)
    assert ax is not None
    # Add more assertions if needed

def test_plot_top_loadings(sample_model):
    # Test plot_top_loadings function
    ax = plot_top_loadings(sample_model, 0, "factor1")
    assert ax is not None
    # Add more assertions if needed

def test_plot_variance_explained(sample_model):
    # Test plot_variance_explained function
    ax = plot_variance_explained(sample_model)
    assert ax is not None
    # Add more assertions if needed

def test_plot_variance_explained_factor(sample_model):
    # Test plot_variance_explained_factor function
    ax = plot_variance_explained_factor(sample_model)
    assert ax is not None
    # Add more assertions if needed

def test_plot_variance_explained_view(sample_model):
    # Test plot_variance_explained_view function
    ax = plot_variance_explained_view(sample_model)
    assert ax is not None
    # Add more assertions if needed

def test_plot_factor_covariate_cor(sample_model):
    # Test plot_factor_covariate_cor function
    ax = plot_factor_covariate_cor(sample_model, [0, 1, 2])
    assert ax is not None
    # Add more assertions if needed

def test_plot_fit(sample_model):
    # Test plot_fit function
    ax = plot_fit(sample_model, 0)
    assert ax is not None
    # Add more assertions if needed

def test_plot_enrichment():
    # Test plot_enrichment function
    gene_list = ["gene1", "gene2", "gene3"]
    background = ["gene1", "gene2", "gene3", "gene4", "gene5"]
    db = ["database1", "database2"]
    top_n = [5, 3]
    ax = plot_enrichment(gene_list, background, db, top_n)
    assert ax is not None
    # Add more assertions if needed