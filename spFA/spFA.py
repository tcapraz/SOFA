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

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)


def npsigmoid(x):
    return 1 / (1 + np.exp(-x))


class spFA:
    def __init__(self, X, llh, num_factors, views, y=None, target_llh=None, supervised_factors=0, device=torch.device("cpu"), ard=True, horseshoe=True):
        """


        Parameters
        ----------
        X : list of torch tensors
            list of input data views. each view should be centered and scaled.
        llh : list of str
            likelihoods for views in X
        num_factors : int
            number of latent factors
        views : list of str
            names of the input data views in X
        y : torch tensor, optional
            optional target data. The default is None.
        target_llh : str, optional
            likelihood for target data y. The default is None.
        supervised_factors : int, optional
            number of factors that should be supervised by y. The default is 0.
        device : str, optional
            device to fit the model ("cuda" or "cpu").
            The default is torch.device('cpu').
        ard : bool, optional
            whether to use ARD priors on the loadings.
            The default is True.
        horseshoe : bool, optional
            whethere to use horseshoe priors on the loadings.
            The default is True.
        """

        self.num_factors = num_factors
        self.supervised_factors = supervised_factors

        self.device = device
        self.isfit = False
        self.X = [i.to(device) for i in X]
        self.llh = llh
        self.y = y.to(device)
        self.target_llh = target_llh
        self.views = views
        self.ard = ard
        self.horseshoe = horseshoe
        self.history = []

        if self.target_llh == "multinomial":
            self.k = len(np.unique(y.numpy()))

    def sFA_model(self):
        X = self.X
        y = self.y
        llh = self.llh

        device = self.device
        num_factors = self.num_factors
        supervised_factors = self.supervised_factors
        num_views = len(X)

        num_samples = X[0].shape[0]
        num_features = [i.shape[1] for i in X]

        sigma_data = pyro.param("sigma_data", torch.ones(1, device=device), constraint=pyro.distributions.constraints.positive)
        sigma_response = pyro.param("sigma_response", torch.ones(1, device=device), constraint=pyro.distributions.constraints.positive)
        if self.horseshoe:
            tau = pyro.sample("tau", dist.HalfCauchy(torch.ones(1, device=device)))

        W = []
        for i in range(num_views):
            with pyro.plate("factors_{}".format(i), num_factors):
                if self.ard:
                    W_scale = pyro.sample("ard_prior_{}".format(i), dist.Gamma(torch.ones(num_features[i], device=device), torch.ones(num_features[i], device=device)).to_event(1))
                    W_ = pyro.sample("W_unshrunk_{}".format(i), dist.Normal(torch.zeros(num_features[i], device=device), 1 / W_scale).to_event(1))
                else:
                    W_ = pyro.sample("W_unshrunk_{}".format(i), dist.Normal(torch.zeros(num_features[i], device=device), torch.ones(num_features[i], device=device)).to_event(1))
                if self.horseshoe:
                    lam = pyro.sample("lam_{}".format(i), dist.HalfCauchy(torch.ones(num_features[i], device=device)).to_event(1))
                    W_ = pyro.deterministic("W_{}".format(i), W_ * lam**2 * tau**2)
                else:
                    W_ = pyro.deterministic("W_{}".format(i), W_)

            W.append(W_)

        if supervised_factors > 0:
            with pyro.plate("betas", supervised_factors):
                if self.target_llh == "multinomial":
                    beta = pyro.sample("beta", dist.Normal(torch.zeros(self.k, device=device), torch.ones(self.k, device=device)).to_event(1))

                else:
                    beta = pyro.sample("beta", dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)))

        with pyro.plate("data", num_samples):
            Z = pyro.sample("Z", dist.Normal(torch.zeros(num_factors, device=device), torch.ones(num_factors, device=device)).to_event(1))

            X_pred = []
            for i in range(num_views):
                if llh[i] == "bernoulli":
                    X_pred.append(Z @ W[i])
                    pyro.sample("obs_data_{}".format(i), dist.Bernoulli(sigmoid(X_pred[i])).to_event(1), obs=X[i])
                else:
                    for j in range(num_factors):
                        X_ij = pyro.deterministic(f"X_{i}{j}", Z[:, [j]] @ W[i][[j], :])
                    X_i = pyro.deterministic(f"X_{i}", Z @ W[i])

                    X_pred.append(X_i)

                    pyro.sample("obs_data_{}".format(i), dist.Normal(X_i, sigma_data).to_event(1), obs=X[i])

            if supervised_factors > 0:
                y_pred = Z[:, 0 : self.supervised_factors] @ beta
                if self.target_llh == "bernoulli":
                    pyro.sample("obs_response", dist.Bernoulli(sigmoid(y_pred)).to_event(1), obs=y)

                elif self.target_llh == "multinomial":
                    y_pred = softmax(y_pred)
                    pyro.sample("obs_response", dist.Categorical(y_pred).to_event(1), obs=y)
                else:
                    pyro.sample("obs_response", dist.Normal(y_pred, sigma_response).to_event(1), obs=y)

    def sFA_guide(self):
        X = self.X
        y = self.y
        llh = self.llh

        device = self.device
        num_factors = self.num_factors
        supervised_factors = self.supervised_factors

        num_views = len(X)
        num_samples = X[0].shape[0]
        num_features = [i.shape[1] for i in X]

        Z_loc = pyro.param("Z_loc", torch.zeros((num_samples, num_factors), device=device))
        Z_scale = pyro.param("Z_scale", torch.ones((num_samples, num_factors), device=device), constraint=pyro.distributions.constraints.positive)

        if supervised_factors > 0:
            if self.target_llh == "multinomial":
                beta_loc = pyro.param("beta_loc", torch.zeros(supervised_factors, self.k, device=device))
                beta_scale = pyro.param("beta_scale", torch.ones(supervised_factors, self.k, device=device), constraint=pyro.distributions.constraints.positive)
                with pyro.plate("betas", supervised_factors):
                    beta = pyro.sample("beta", dist.Normal(beta_loc, beta_scale).to_event(1))
            else:
                beta_loc = pyro.param("beta_loc", torch.zeros(supervised_factors, device=device))
                beta_scale = pyro.param("beta_scale", torch.ones(supervised_factors, device=device), constraint=pyro.distributions.constraints.positive)
                with pyro.plate("betas", supervised_factors):
                    beta = pyro.sample("beta", dist.Normal(beta_loc, beta_scale))

        if self.horseshoe:
            tau_loc = pyro.param("tau_loc", torch.ones(1, device=device), constraint=dist.constraints.positive)
            tau = pyro.sample("tau", dist.Delta(tau_loc))
            lam_loc = []
        if self.ard:
            gamma_alpha = []
            gamma_beta = []
        W_loc = []
        W_scale = []

        for i in range(num_views):
            if self.ard:
                gamma_alpha.append(pyro.param("gamma_alpha_{}".format(i), torch.ones((num_factors, num_features[i]), device=device), constraint=dist.constraints.positive))
                gamma_beta.append(pyro.param("gamma_beta_{}".format(i), torch.ones((num_factors, num_features[i]), device=device), constraint=dist.constraints.positive))
            else:
                W_scale.append(pyro.param("W_scale_{}".format(i), torch.ones((num_factors, num_features[i]), device=device), constraint=pyro.distributions.constraints.positive))

            W_loc.append(pyro.param("W_loc_{}".format(i), torch.zeros((num_factors, num_features[i]), device=device)))
            if self.horseshoe:
                lam_loc.append(pyro.param("lam_loc_{}".format(i), torch.ones((num_factors, num_features[i]), device=device), constraint=dist.constraints.positive))

            with pyro.plate("factors_{}".format(i), num_factors):
                if self.ard:
                    W_scale = pyro.sample("ard_prior_{}".format(i), dist.Delta(gamma_alpha[i] / gamma_beta[i]).to_event(1))
                    W = pyro.sample("W_unshrunk_{}".format(i), dist.Normal(W_loc[i], 1 / W_scale).to_event(1))
                else:
                    W = pyro.sample("W_unshrunk_{}".format(i), dist.Normal(W_loc[i], W_scale[i]).to_event(1))
                if self.horseshoe:
                    lam = pyro.sample("lam_{}".format(i), dist.Delta(lam_loc[i]).to_event(1))

        with pyro.plate("data", num_samples):
            pyro.sample("Z", dist.Normal(Z_loc, Z_scale).to_event(1))

    def fit_spFA(self, n_steps=3000, lr=0.005, refit=False):
        """
        method to fit the spFA model

        Parameters
        ----------
        n_steps : int, optional
            number of iterations for fitting. The default is 3000.
        lr : float, optional
            learning rate for adam optimizer. The default is 0.005.
        refit : bool, optional
            whether to refit the model. the default behaviour is that
            the model will not be newly intialized if you call fit_spFA
            twice with refit=False.
            The default is False.

        Returns
        -------
        None.
        """

        adam_params = {"lr": lr, "betas": (0.95, 0.999)}
        optimizer = Adam(adam_params)

        if self.isfit == False or refit:
            pyro.clear_param_store()
            self.svi = SVI(self.sFA_model, self.sFA_guide, optimizer, loss=Trace_ELBO())

        pbar = tqdm(range(n_steps))
        # do gradient steps
        for step in pbar:
            loss = self.svi.step()
            # track loss
            self.history.append(loss)

            if step % 1000 == 0:
                pbar.set_description(f"Current Elbo {loss:.2E}")

        print("training done")
        self.isfit = True

    def get_factors(self):
        """
        Sample factors from posterior

        Returns
        -------
        Z_pred : numpy array
        """
        predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=["Z"])
        samples = predictive()
        Z_pred = np.mean(samples["Z"].cpu().numpy(), axis=0)
        return Z_pred

    def get_loadings(self, view=None):
        """
        Sample loadings from posterior

        Parameters
        ----------
        view : int, optional
            index of view to get loadings for.
            if view is None loadings from all views are returned.
            The default is None.

        Returns
        -------
        W_pred: list of numpy arrays or numpy array
        """
        if view == None:
            W_pred = []
            for i in range(len(self.X)):
                predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"W_{i}"])
                samples = predictive()
                W_pred.append(np.mean(samples[f"W_{i}"].cpu().numpy(), axis=0))
            return W_pred
        else:
            predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"W_{view}"])
            samples = predictive()
            return np.mean(samples[f"W_{view}"].cpu().numpy(), axis=0)

    def get_Xpred(self, view=None):
        """
        Sample predicted X from posterior.

        Parameters
        ----------
        view : int, optional
            index of view to get predicted X for.
            if view is None all predicted views are returned.
            The default is None.

        Returns
        -------
        X_pred : list of numpy arrays or numpy array
        """
        params = {i: j for i, j in pyro.get_param_store().items()}

        if view == None:
            X_pred = []
            for i in range(len(self.X)):
                predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"X_{i}"])
                samples = predictive()

                X_pred.append(np.mean(samples[f"X_{i}"].cpu().numpy(), axis=0))
            return X_pred
        else:
            predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"X_{view}"])
            samples = predictive()
            X_pred = np.mean(samples[f"X_{view}"].cpu().numpy(), axis=0)
            return X_pred

    def get_Xpred_perfactor(self, view=None):
        """
        Sample predicted X  for each factor from posterior.
        (used to compute explained variance per factor)

        Parameters
        ----------
        view : int, optional
            index of view to get predicted X for.
            if view is None all predicted views are returned.
            The default is None.

        Returns
        -------
        X_pred : list of numpy arrays
        """
        params = {i: j for i, j in pyro.get_param_store().items()}

        if view is None:
            X_pred = []
            for i in range(len(self.X)):
                for j in range(len(self.num_factors)):
                    predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"X_{i}{j}"])
                    samples = predictive()
                    X_pred.append(np.mean(samples[f"X_{i}{j}"].cpu().numpy(), axis=0))
            return X_pred
        else:
            X_pred = []
            for j in range(self.num_factors):
                predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"X_{view}{j}"])
                samples = predictive()
                X_pred.append(np.mean(samples[f"X_{view}{j}"].cpu().numpy(), axis=0))
            return X_pred

    def get_beta(self):
        """
        get fitted beta coefficients of regression target y

        Returns
        -------
        beta: numpy array
        """
        params = {i: j for i, j in pyro.get_param_store().items()}

        return params["beta_loc"].detach().cpu().numpy()
