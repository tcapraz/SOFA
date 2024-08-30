import pytest
from sofa.models.SOFA import SOFA
import torch
from muon import MuData
import numpy as np

def test_SOFA_initialization():
    Xmdata = None
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

    model = SOFA(Xmdata, num_factors, Ymdata, design, device, horseshoe, update_freq, subsample, metadata, verbose, horseshoe_scale_feature, horseshoe_scale_factor, horseshoe_scale_global, seed)

    assert model.num_factors == num_factors
    assert model.supervised_factors == 0
    assert model.Xmdata == Xmdata
    assert model.Ymdata == Ymdata
    assert model.design == design
    assert model.device == device
    assert model.horseshoe == horseshoe
    assert model.update_freq == update_freq
    assert model.subsample == subsample
    assert model.metadata == metadata
    assert model.verbose == verbose
    assert model.horseshoe_scale_feature == horseshoe_scale_feature
    assert model.horseshoe_scale_factor == horseshoe_scale_factor
    assert model.horseshoe_scale_global == horseshoe_scale_global
    assert model.seed == seed

def test_SOFA_fit(sample_data):
    Xmdata = MuData({"m1": sample_data, "m2": sample_data})
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

    model = SOFA(Xmdata, num_factors, Ymdata, design, device, horseshoe, update_freq, subsample, metadata, verbose, horseshoe_scale_feature, horseshoe_scale_factor, horseshoe_scale_global, seed)

    model.fit(n_steps=100, lr=0.01)

    assert len(model.history) == 100

def test_SOFA_predict(sample_data):
    Xmdata = MuData({"m1": sample_data, "m2": sample_data})
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

    model = SOFA(Xmdata, num_factors, Ymdata, design, device, horseshoe, update_freq, subsample, metadata, verbose, horseshoe_scale_feature, horseshoe_scale_factor, horseshoe_scale_global, seed)

    model.fit(n_steps=100, lr=0.01)

    Z_pred = model.predict("Z")
    W_pred = model.predict("W")

    assert Z_pred.shape == (model.num_samples, model.num_factors)
    assert W_pred[0].shape == (model.num_factors, sample_data.shape[1])

def test_SOFA_save_as_mudata(sample_data):
    Xmdata = MuData({"m1": sample_data, "m2": sample_data})
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

    model = SOFA(Xmdata, num_factors, Ymdata, design, device, horseshoe, update_freq, subsample, metadata, verbose, horseshoe_scale_feature, horseshoe_scale_factor, horseshoe_scale_global, seed)

    model.fit(n_steps=100, lr=0.01)

    mdata = model.save_as_mudata()

    assert isinstance(mdata, MuData)


def test_SOFA_simulate():
    num_views = 2
    num_features = [10, 20]
    num_samples = 100
    num_factors = 2
    llh = ["gaussian", "gaussian"]
    num_guide_views = 1
    sigma_response = [0.1]
    guide_llh = ["gaussian"]
    design = torch.tensor([[1, 0], [0, 0]])
    k = [2]
    y_dim = [1]

    model = SOFA()
    X, Y, W, Z, beta, beta0 = model._simulate(sigma_data=1, 
                                              num_views=num_views,
                                              num_features=num_features, 
                                              num_samples=num_samples, 
                                              num_factors=num_factors, 
                                              llh=llh, 
                                              num_guide_views=num_guide_views, 
                                              sigma_response=sigma_response, 
                                              guide_llh=guide_llh, 
                                              design=design, 
                                              return_data=True, 
                                              k=k, 
                                              y_dim=y_dim)

    assert len(X) == num_views
    assert len(Y) == num_guide_views
    assert len(W) == num_views
    assert Z.shape == (num_samples, num_factors)
    assert len(beta) == num_guide_views
    assert len(beta0) == num_guide_views