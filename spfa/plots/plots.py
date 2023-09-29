from typing import Union, List
import pyro
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import scipy.stats as stats
from ..utils.utils import calc_var_explained, calc_var_explained_view
from ..models.spFA import spFA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes 

def plot_weights(
        model: spFA, 
        view: str, 
        factor: int=0
        ) -> Axes:
    """
    Plot the weights of a specific factor for a given view in the spFA model.

    Parameters
    ----------
    model : spFA
        The spFA model to plot the weights for.
    view : str
        The name of the view to plot the weights for.
    factor : int, optional
        The index of the factor to plot the weights for. Default is 0.

    Returns
    -------
    matplotlib Axes object
        The plot of the weights for the specified factor and view.
    """
    
    if not hasattr(model, f"W"):
        model.W = [model.predict(f"W_{i}", num_split=10000) for i in range(len(model.X))] 
    W = pd.DataFrame(model.W[view], columns=model.Xmdata.mod[list(model.Xmdata.mod.keys())[view]].var_names)
    W = W.loc[factor, :]
    W_sorted = W.sort_values()
    labels = np.array(W_sorted.index.tolist())
    labels[10:len(labels)-10] = ""
    x = [i for i in range(len(W_sorted))]
    fig, ax = plt.subplots(1)
    ax.scatter(x, W_sorted)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], W_sorted[i]))
    return ax


def plot_top_weights(
        model: spFA,
        view: int,
        factor: int,
        top_n:  int=10,
        sign: Union[None,str]=None, 
        highlight: Union[None,str]=None
        ) -> Axes:
    """
    Plot the top weights of a factor in a given view.

    Parameters
    ----------
    model : spFA
        The trained model object.
    view : int
        The index of the view to plot.
    factor : str
        The name of the factor to plot.
    top_n : int, optional
        The number of top weights to plot, by default 10.
    sign : str, optional
        The sign of the weights to plot. If None, plot the top absolute weights.
        If "+" or "-", plot the top positive or negative weights, respectively.
    highlight : list, optional
        A list of feature names to highlight in red, by default None.

    Returns
    -------
    matplotlib  Axes object
        The plot of the top weights.
    """
    if not hasattr(model, f"W"):
        model.W = [model.predict(f"W_{i}", num_split=10000) for i in range(len(model.X))]
        
    W = pd.DataFrame(model.W[view], columns = model.Xmdata.mod[list(model.Xmdata.mod.keys())[view]].var_names)
    
    W = W.loc[factor,:]

    if sign ==None:
        idx = np.argpartition(abs(W), -top_n)[-top_n:]

        topW = W.iloc[idx]
        order = topW.map(lambda x : x).sort_values(ascending = True)
        topW = topW[order.index]
    else:
        if sign=="-":
            W = W*-1
        order = W.map(lambda x : x).sort_values(ascending = False)
        topW = W[order.index][0:top_n]
        topW = topW.iloc[::-1]
        
    if highlight !=None:
        highlight_id= [i for i, e in enumerate(topW.index.tolist()) if e in highlight]
    my_range=range(1,len(topW.index)+1)
 
    # The horizontal plot is made using the hline() function
    fig, ax = plt.subplots(1)
    ax.hlines(y=my_range, xmin=0, xmax=topW,alpha=0.4)
    ax.scatter(topW, my_range, alpha=1)
     
    # Add title and axis names
    ax.set_yticks(my_range, topW.index)
    signoffset = max(topW)/50
    if sign == "-":
        for i in range(len(topW)):
            ax.annotate("-", (topW[i]+signoffset,i+0.7), fontsize=16)
    if sign == "+":
        for i in range(len(topW)):
            ax.annotate("+", (topW[i]+signoffset,i+0.7), fontsize=12)
    ax.set_xlabel('Loadings')
    ax.set_ylabel('Features')
    if highlight != None:
        for j in highlight_id:
            ax.setp(ax.get_yticklabels()[j], color='red')

    return ax

def plot_variance_explained(model: spFA) -> Axes:
    """
    Plot the variance explained by each factor for each view.

    Parameters
    ----------
    model : spFA
        The trained spFA model.

    Returns
    -------
    matplotlib Axes object
        The plot of the variance explained by each factor for each view.
    """
    X = [i.cpu().numpy() for i in model.X]
    vexp = []
    if not hasattr(model, "Z"):
        model.Z = model.predict("Z", num_split=10000)
    if not hasattr(model, f"W"):
        model.W = [model.predict(f"W_{i}", num_split=10000) for i in range(len(X))]   
    for i in range(len(X)):
        mask = model.Xmask[i].cpu().numpy()
        X_pred_factor = []
        for j in range(model.num_factors):
            X_pred_factor.append(model.Z[mask,j, np.newaxis] @ model.W[i][np.newaxis,j,:])
        vexp.append(calc_var_explained(X_pred_factor, X[i][mask,:]).reshape(model.num_factors,1))
    vexp = np.hstack(vexp)
    if model.Ymdata is not None:
        y_labels = np.array(["" for i in range(model.num_factors)], dtype=object)
        guided_factors = list(model.Ymdata.mod.keys())
        for i in range(len(guided_factors)):
            s =  "(" + guided_factors[i] + ")"
            y_labels[model.design.cpu().numpy()[i,:]==1] =s

        for i in range(model.num_factors):
            y_labels[i] = y_labels[i] + f" {i+1}"
    else:
        y_labels = np.arange(1,model.num_factors+1)
    fig, ax = plt.subplots(1)
    plot = ax.imshow(vexp, cmap="Blues", origin="lower")
    ax.set_xlabel("View")
    ax.set_xticks(ticks = range(len(model.views)), labels= model.views,rotation=90)
    ax.set_yticks(ticks = range(vexp.shape[0]), labels= y_labels)
    ax.set_ylabel("Factor")
    plt.colorbar(plot)
    return ax

def plot_variance_explained_factor(model: spFA) -> Axes:

    """Plots the variance explained by each factor.

    Parameters
    ----------
    model : spFA
        The trained spFA model.

    Returns
    -------
    matplotlib Axes object
        The plot axes.
    """    
 
    X = [i.cpu().numpy() for i in model.X]
    vexp = []
    if not hasattr(model, "Z"):
        model.Z = model.predict("Z", num_split=10000)
    if not hasattr(model, f"W"):
        model.W = [model.predict(f"W_{i}", num_split=10000) for i in range(len(X))]   
    for i in range(len(X)):
        mask = model.Xmask[i].cpu().numpy()
        X_pred_factor = []
        for j in range(model.num_factors):
            X_pred_factor.append(model.Z[mask,j, np.newaxis] @ model.W[i][np.newaxis,j,:])
        vexp.append(calc_var_explained(X_pred_factor, X[i][mask,:]).reshape(model.num_factors,1))
    vexp = np.hstack(vexp)
    vexp = np.sum(vexp, axis=1)

    fig, ax = plt.subplots(1)
    plot = ax.bar(["Factor"+ str(i+1) for  i in range(len(vexp))], vexp)
    ax.set_xlabel("View")
    ax.set_ylabel("R2")
    ax.set_xticks(ticks = range(len(vexp)),rotation=90)
    return ax

def plot_variance_explained_view(model: spFA) -> Axes:
    """Plots the variance explained of each view.

    Parameters
    ----------
    model : spFA
        The trained spFA model.

    Returns
    -------
    Axes
        The plot axes.
    """    
    X = [i.cpu().numpy() for i in model.X]

    vexp = []
    if not hasattr(model, f"X_pred"):
        model.X_pred = [model.predict(f"X_{i}", num_split=10000) for i in range(len(X))]
        
    for i in range(len(X)):
        mask = model.Xmask[i].cpu().numpy()
        vexp.append(calc_var_explained_view(model.X_pred[i][mask,:], X[i][mask,:]))

    fig, ax = plt.subplots(1)
    plot = ax.bar(["view"+ str(i) for  i in range(len(vexp))], vexp)
    ax.set_xlabel("View")
    ax.set_ylabel("R2")
    ax.set_xticks(ticks = range(len(model.views)), labels= model.views,rotation=90)
    return ax

def plot_factor_covariate_cor(
        model: spFA,
        metavar: List[int]
        ) -> Axes:
    """
    Plot the correlation between the factors and covariates.

    Parameters
    ----------
    model : spFA
        The trained spFA model.
    metavar : list of str
        The list of covariate names to plot

    Returns
    -------
    matplotlib Axes object
        Heatmap with the correlation between the factors and covariates.
    """
    cor = []
    if not hasattr(model, "Z"):
        model.Z = model.predict("Z", num_split=10000)
    for i in metavar:
        var = model.metadata[i].values
        mask = ~model.metadata[i].isna()
        if var.dtype.type is np.string_ or var.dtype.type is np.object_ or var.dtype.type is pd.core.dtypes.dtypes.CategoricalDtypeType:
            lmap = {l:j for j,l in enumerate(np.unique(var[mask]))}
            y = np.array([lmap[l] for l in var[mask]])
        else:
            y = var[mask]
        cor.append([stats.pearsonr(model.Z[mask,z], y)[0] for z in range(model.Z.shape[1])])

    if model.Ymdata is not None:

        Ymdata = model.Ymdata

        y_labels = model.Ymdata.mod
        y_labels = ["(" +i +") " for i in y_labels]
        y_labels = y_labels + ["" for i in range(model.Z.shape[1] - len(y_labels))]
        y_labels = [i+ str(idx+1) for idx,i in enumerate(y_labels)]
    else:
        y_labels = np.arange(1,model.num_factors+1)
    cormat = np.stack(cor)
    cormat = cormat.T

    fig, ax = plt.subplots(1)
    plot = ax.imshow(abs(cormat), cmap="Oranges", origin="lower")
    ax.xlabel("Covariate")
    ax.set_xticks(ticks = range(len(metavar)), labels= metavar, rotation=90)
    ax.set_yticks(ticks = range(cormat.shape[0]), labels= y_labels)
    ax.set_ylabel("Factor")
    plt.colorbar(plot)
    return ax

def plot_fit(
        model: spFA,
        view: int
        ) -> Axes:
    """
    Plot the reconstructions of data.

    Parameters
    ----------
    model : spFA
        The trained spFA model.
    view : int
        The index of the view to plot.

    Returns
    -------
    matplotlib Axes object
        The plot of the reconstructions.
    """
    X = model.X[view].cpu().numpy()
    X_pred = model.X_pred[view]

    fig, ax = plt.subplots(1)
    ax.scatter(X, X_pred, alpha=0.2, s=1)
    ax.plot([np.min(X_pred), np.max(X_pred)], [np.min(X_pred), np.max(X_pred)], color = "black")
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("X predicted")
    return ax

