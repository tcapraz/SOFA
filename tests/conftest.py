import pytest
from sofa.models.SOFA import SOFA
import torch
import sofa
import numpy as np
import pandas as pd
from muon import MuData

@pytest.fixture(scope="session", name="sample_model")
def simulated_model():
    num_views = 2
    num_features = [10, 20]
    num_samples = 100
    num_factors = 2
    llh = ["gaussian", "gaussian"]
    num_guide_views = 1
    sigma_response = [0.1]
    guide_llh = ["gaussian"]
    design = torch.tensor([[1, 0], [0, 0]])
    k = [1]
    y_dim = [1]
    sigma_data = [1,1]
    model = SOFA()
    X, Y, W, Z, beta, beta0, lam_feature, tau = model._simulate(sigma_data=sigma_data, 
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
    model.Xmdata = MuData({"view1": sofa.tl.get_ad(pd.DataFrame(X[0])), "view2": sofa.tl.get_ad(pd.DataFrame(X[1]))})
    model.Ymdata = MuData({"guide1": sofa.tl.get_ad(pd.DataFrame(Y[0]))})

    model.X = [torch.tensor(i) for i in X]
    model.Y = [torch.tensor(i) for i in Y]
    model.W = W
    model.Z = Z
    model.beta = beta
    model.beta0 = beta0
    model.lam_feature = lam_feature
    model.tau = tau
    model.isfit = True
    model.views = ["view1", "view2"]
    model.metadata = pd.DataFrame(np.random.normal(0,1,(num_samples, 5)), columns=[f"covariate_{i}" for i in range(5)])
    return model

@pytest.fixture(scope="session", name="sample_data")
def sample_data():
    x = np.random.normal(0,1,(10,10))
    data = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])
    return sofa.tl.get_ad(data)