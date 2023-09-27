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

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)


def npsigmoid(x):
    return 1 / (1 + np.exp(-x))


class spFA:
    def __init__(self, Xmdata,  num_factors, Ymdata=None,  target_llh=None, design=None, device=torch.device("cpu"), ard=True, horseshoe=True, update_freq=200, subsample=0, metadata=None, target_scale=1000):
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
        update_freq: int
            Frequency of steps before ELBO is displayed during training.
        """

        self.num_factors = num_factors

        self.device = device
        self.isfit = False
        self.Xmdata = Xmdata
        self.Ymdata = Ymdata

        self.X, self.views, self.llh, self.Xmask = self._data_handler()
        self.scale = np.ones(len(Xmdata.mod))
        self.ard = ard
        self.horseshoe = horseshoe
        self.history = []
        self.update_freq = update_freq
        self.metadata = metadata
        self.target_scale = target_scale
    
        if Ymdata is not None:
            self.Y, self.target_views, self.target_llh, self.k, self.y_dim, self.Ymask  = self._target_handler()
        else:
            self.Y = None
            self.target_llh = None
            
                
        if design is not None:
            self.design = design.to(device)
            self.supervised_factors = torch.sum(torch.any(design!=0, dim=0))
        else:
            self.supervised_factors = 0
            self.design = None

        self.num_samples = Xmdata.n_obs
        self.subsample = subsample
        self.idx = torch.arange(self.num_samples)
    
    def _data_handler(self):
        views = []
        X = []
        llh = []
        mask = []
        for i in self.Xmdata.mod:
            views.append(i)
            data = self.Xmdata.mod[i].X.copy()
            data[np.isnan(data)] = 0
            #data = data - np.mean(data[self.Xmdata.mod[i].obsm["mask"]], axis=0)
            X.append(torch.tensor(data).to(self.device))
            llh.append(self.Xmdata.mod[i].uns["llh"])
            mask.append(torch.tensor(self.Xmdata.mod[i].obsm["mask"]).to(self.device))
        return X, views, llh, mask
    
    def _target_handler(self):
        target_views = []
        Y = []
        target_llh = []
        k = []
        y_dim = []
        mask = []
        self.le = []
        for ix, i in enumerate(self.Ymdata.mod):
            target_views.append(i)
            data = self.Ymdata.mod[i].X.copy()
            data[np.isnan(data)] = 0

            y = torch.tensor(data).to(self.device)
            if y.dtype == np.str_:
                self.le.append(preprocessing.LabelEncoder())
                y = torch.tensor(self.le[ix].fit_transform(y)).to(self.device)
            t_llh = self.Ymdata.mod[i].uns["llh"]
            target_llh.append(t_llh)

            mask.append(torch.tensor(self.Ymdata.mod[i].obsm["mask"]).to(self.device))
            if t_llh == "multinomial":
                k.append(len(np.unique(y.cpu().numpy())))
                Y.append(y.flatten())
            else:
                k.append(1)
                Y.append(y)

            if len(y.shape) > 1:
                y_dim.append(y.shape[1])
            else:
                y_dim.append(1)
            
        
        return Y, target_views, target_llh, k, y_dim, mask
    
    def sFA_model(self, idx, subsample=32):
        X = self.X
        Y = self.Y
        llh = self.llh
        target_llh = self.target_llh
        design = self.design
        
        device = self.device
        num_factors = self.num_factors
        supervised_factors = self.supervised_factors
        num_views = len(X)
        if Y != None:
            num_target_views = len(Y)
        
        num_samples = X[0].shape[0]
        num_features = [i.shape[1] for i in X]

        sigma_data = pyro.param("sigma_data", torch.ones(num_views, device=device), constraint=pyro.distributions.constraints.positive)
        if Y != None:
            sigma_response = pyro.param("sigma_response", torch.ones(num_target_views, device=device), constraint=pyro.distributions.constraints.positive)

        if self.horseshoe:
            tau = pyro.sample("tau", dist.HalfCauchy(torch.ones(1, device=device)))

        W = []
        #means = []
        for i in range(num_views):
            #means.append(pyro.param(f"mean_{i}", torch.zeros(num_features[i], device=device)))
            with pyro.plate("factors_{}".format(i), num_factors):
                if self.ard:
                    W_scale = pyro.sample("ard_prior_{}".format(i), dist.Gamma(torch.ones(num_features[i], device=device), torch.ones(num_features[i], device=device)).to_event(1))
                    W_ = pyro.sample("W_unshrunk_{}".format(i), dist.Normal(torch.zeros(num_features[i], device=device), 1 / W_scale).to_event(1))
                else:
                    W_ = pyro.sample("W_unshrunk_{}".format(i), dist.Normal(torch.zeros(num_features[i], device=device), torch.ones(num_features[i], device=device)).to_event(1))
                if self.horseshoe:
                    lam = pyro.sample("lam_{}".format(i), dist.HalfCauchy(torch.ones(num_features[i], device=device)*10).to_event(1))
                    W_ = pyro.deterministic("W_{}".format(i), W_ * lam**2 * tau**2)
                else:
                    W_ = pyro.deterministic("W_{}".format(i), W_)

            W.append(W_)

        if supervised_factors > 0:
            beta = []
            for i in range(num_target_views):
                with pyro.plate(f"betas_{i}", int(torch.sum(design[i,:]))):
                    if target_llh[i] == "multinomial":
                        beta_ = pyro.sample(f"beta_{i}", dist.Normal(torch.zeros(self.k[i], device=device), torch.ones(self.k[i], device=device)).to_event(1))
                    else:
                        beta_ = pyro.sample(f"beta_{i}", dist.Normal(torch.zeros(self.y_dim[i], device=device), torch.ones(self.y_dim[i], device=device)).to_event(1))
                beta.append(beta_)
              
        if subsample > 0:
            data_plate = pyro.plate("data", num_samples, subsample_size=subsample)
        else:
            data_plate = pyro.plate("data", num_samples, subsample=idx)
            
        with data_plate as ind:
            ind = ind.flatten()
            Z = pyro.sample("Z", dist.Normal(torch.zeros(num_factors, device=device), torch.ones(num_factors, device=device)).to_event(1))
            x_ = torch.cat(X, dim=1).float()
            w_ = torch.cat(W, dim=1)

            Z_pred = pyro.deterministic(f"Z_pred", x_ @ w_.T)

            X_pred = []
            for i in range(num_views):

                with pyro.poutine.scale(scale=self.scale[i]):
                    with pyro.poutine.mask(mask=self.Xmask[i][ind]):
                        if llh[i] == "bernoulli":
                            X_pred.append(Z @ W[i])
                            X_i = pyro.deterministic(f"X_{i}", sigmoid(Z @ W[i]))
                            for j in range(num_factors):
                                X_ij = pyro.deterministic(f"X_{i}{j}", sigmoid(Z[:, [j]] @ W[i][[j], :]))
                            pyro.sample("obs_data_{}".format(i), dist.Bernoulli(sigmoid(X_pred[i])).to_event(1), obs=X[i][ind,:])
                        else:
                            for j in range(num_factors):
                                X_ij = pyro.deterministic(f"X_{i}{j}", Z[:, [j]] @ W[i][[j], :])
                            X_i = pyro.deterministic(f"X_{i}", Z @ W[i])
                            X_pred.append(X_i)
                            pyro.sample("obs_data_{}".format(i), dist.Normal(X_i, sigma_data[i]).to_event(1), obs=X[i][ind,:])
            if supervised_factors > 0:
                for i in range(num_target_views):
                    with pyro.poutine.scale(scale=self.target_scale[i]):
                        with pyro.poutine.mask(mask=self.Ymask[i][ind]):
                            y_pred = Z[:, design[i,:]==1] @ beta[i]
                            pyro.deterministic(f"Y_{i}", y_pred)
                            if target_llh[i] == "gaussian":
                                pyro.sample(f"obs_response_{i}", dist.Normal(y_pred, sigma_response[i]).to_event(1), obs=Y[i][ind])
                            #if target_llh[i] == "gamma":
                             #   shape = sigma_response[i]*y_pred
                                #shape = shape.clamp(min=torch.finfo(torch.float32).eps)
                                #scale = torch.exp(y_pred)*torch.sqrt(torch.tensor(np.pi))/torch.sqrt(torch.tensor(2))
                            #    pyro.sample(f"obs_response_{i}", dist.Gamma(torch.exp(shape),sigma_response[i]).to_event(1), obs=Y[i][ind])
                            elif target_llh[i] == "bernoulli":
                                pyro.sample(f"obs_response_{i}", dist.Bernoulli(sigmoid(y_pred)).to_event(1), obs=Y[i][ind])
                            elif target_llh[i] == "multinomial":
                                pyro.sample(f"obs_response_{i}", dist.Categorical(softmax(y_pred)).to_event(1), obs=Y[i][ind])

    def sFA_guide(self, idx, subsample=32):
        X = self.X
        Y = self.Y
        llh = self.llh

        device = self.device
        num_factors = self.num_factors
        supervised_factors = self.supervised_factors
        design = self.design
                            
        num_views = len(X)
        if Y != None:
            num_target_views = len(Y)
        num_samples = X[0].shape[0]
        num_features = [i.shape[1] for i in X]

        Z_loc = pyro.param("Z_loc", torch.zeros((num_samples, num_factors), device=device))
        Z_scale = pyro.param("Z_scale", torch.ones((num_samples, num_factors), device=device), constraint=pyro.distributions.constraints.positive)
        
        beta_loc = []
        beta_scale = []
                            
        if supervised_factors > 0:
            for i in range(num_target_views):
                if self.target_llh[i] == "multinomial":
                    beta_loc.append(pyro.param(f"beta_loc_{i}", torch.zeros(int(torch.sum(design[i,:])), self.k[i], device=device)))
                    beta_scale.append(pyro.param(f"beta_scale_{i}", torch.ones(int(torch.sum(design[i,:])), self.k[i], device=device), constraint=pyro.distributions.constraints.positive))
                    with pyro.plate(f"betas_{i}", int(torch.sum(design[i,:]))):
                        beta = pyro.sample(f"beta_{i}", dist.Normal(beta_loc[i], beta_scale[i]).to_event(1))
                else:
                    beta_loc.append(pyro.param(f"beta_loc_{i}", torch.zeros(int(torch.sum(design[i,:])), self.y_dim[i], device=device)))
                    beta_scale.append(pyro.param(f"beta_scale_{i}", torch.ones(int(torch.sum(design[i,:])), self.y_dim[i], device=device), constraint=pyro.distributions.constraints.positive))
                    with pyro.plate(f"betas_{i}", int(torch.sum(design[i,:]))):
                        beta = pyro.sample(f"beta_{i}", dist.Normal(beta_loc[i], beta_scale[i]).to_event(1))


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
        if subsample > 0:
            data_plate = pyro.plate("data", num_samples, subsample_size=subsample)
        else:
            data_plate = pyro.plate("data", num_samples, subsample=idx)
            
        with data_plate as ind:
            pyro.sample("Z", dist.Normal(Z_loc[ind,:], Z_scale[ind,:]).to_event(1))
     
    def fit_spFA(self, n_steps=3000, lr=0.005, refit=False, predict=True):
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

        if not self.isfit or refit:
            pyro.clear_param_store()
            self.svi = SVI(self.sFA_model, self.sFA_guide, optimizer, loss=Trace_ELBO())

        pbar = tqdm(range(n_steps))
        last_elbo = np.inf
        # do gradient steps
        for step in pbar:
            loss = self.svi.step(idx=self.idx, subsample=self.subsample)
            # track loss
            self.history.append(loss)

            if step % self.update_freq == 0:
                delta = last_elbo - loss
                pbar.set_description(f"Current Elbo {loss:.2E} | Delta: {delta:.0f}")
                last_elbo = loss

        self.isfit = True
        if predict:
            self.Z = self.predict("Z")
            self.W = []
            self.X_pred = []
            for i in range(len(self.X)):
                self.W.append(self.predict(f"W_{i}"))
                self.X_pred.append(self.predict(f"X_{i}"))
            
        self.rmse=[np.sqrt(np.sum(np.square(self.X_pred[i]-self.X[i].cpu().numpy()))/(self.X_pred[i].shape[0]*self.X_pred[i].shape[1])) for i in range(len(self.X))]
                
    def predict(self, site, num_samples=25, num_split=1024):
        pred = []
        split_obs = torch.split(self.idx, num_split)
        pbar_pred = tqdm(range(len(split_obs)))
        for i in pbar_pred:
            predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=num_samples, return_sites=[site])
            samples = predictive(idx=split_obs[i], subsample=0)
            pred.append(np.mean(samples[site].cpu().numpy(), axis=0))
            torch.cuda.empty_cache()
            pbar_pred.set_description(f"Predicting {site} for obs {torch.min(split_obs[i])}-{torch.max(split_obs[i])}.")
        return np.concatenate(pred)
    
    def predict_pseudoX(self, site):
        pred = []
        split_obs = torch.split(self.idx, self.num_split)
        for i in split_obs:
            predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=25, return_sites=[site], parallel=True)
            samples = predictive(idx=i, subsample=0)
            pred.append(samples[site].cpu().numpy())
        pred = np.concatenate(pred)
        pred = np.mean(pred,axis=0)
        return pred
    
    def get_factors(self):
        """
        Sample factors from posterior

        Returns
        -------
        Z_pred : numpy array
        """
        
        Z_pred = []
        for i in split_obs:
            predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=25, return_sites=["Z"])
            samples = predictive( idx=i, subsample=0)
            
            Z_pred.append(np.mean(samples["Z"].cpu().numpy(), axis=0))
        return np.concatenate(Z_pred)

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
        if view is None:
            W_pred = []
            for i in range(len(self.X)):
                #predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"W_{i}"])
                #samples = predictive(subsample=0)
                #W_pred.append(np.mean(samples[f"W_{i}"].cpu().numpy(), axis=0))
                W_pred.append(self.predict(f"W_{i}"))
            return W_pred
        else:
            predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"W_{view}"])
            samples = predictive(subsample=0)
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

        if view is None:
            X_pred = []
            for i in range(len(self.X)):
                #predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"X_{i}"])
                #samples = predictive(subsample=0)

                #X_pred.append(np.mean(samples[f"X_{i}"].cpu().numpy(), axis=0))
                X_pred.append(self.predict(f"X_{i}"))
            return X_pred
        else:
            #predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"X_{view}"])
            #samples = predictive(subsample=0)
            X_pred = self.predict(f"X_{view}")
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
        #params = {i: j for i, j in pyro.get_param_store().items()}

        if view is None:
            X_pred = []
            for i in range(len(self.X)):
                for j in range(self.num_factors):
                    #predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"X_{i}{j}"])
                    #samples = predictive(subsample=0)
                    #X_pred.append(np.mean(samples[f"X_{i}{j}"].cpu().numpy(), axis=0))
                    X_pred.append(self.predict(f"X_{i}{j}"))

            return X_pred
        else:
            X_pred = []
            for j in range(self.num_factors):
                #predictive = Predictive(self.sFA_model, guide=self.sFA_guide, num_samples=50, return_sites=[f"X_{view}{j}"])
                #samples = predictive(subsample=0)
                #X_pred.append(np.mean(samples[f"X_{view}{j}"].cpu().numpy(), axis=0))
                X_pred.append(self.predict(f"X_{view}{j}", num_split=10000))
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


    
