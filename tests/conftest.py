import pytest
from sofa.models.SOFA import SOFA
import torch

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
    k = [2]
    y_dim = [1]

    model = SOFA()
    X, Y, W, Z, beta, beta0, lam_feature, tau = model._simulate(sigma_data=1, 
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
    model.X = X
    model.Y = Y
    model.W = W
    model.Z = Z
    model.beta = beta
    model.beta0 = beta0
    model.lam_feature = lam_feature
    model.tau = tau
    model.isfit = True
    return model