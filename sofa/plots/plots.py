#!/usr/bin/env python3
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from ..utils.utils import calc_var_explained,  get_gsea_enrichment, get_W, get_var_explained_per_view_factor
import pandas as pd
import numpy as np
from matplotlib.axes import Axes 
from matplotlib import colors
import matplotlib.lines as mlines
import statsmodels
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from ..models.SOFA import SOFA

def plot_loadings(
        model: SOFA, 
        view: str, 
        factor: int=0
        ) -> Axes:
    """
    Plot the loadings of a specific factor for a given view in the SOFA model.

    Parameters
    ----------
    model : SOFA
        The SOFA model to plot the loadings for.
    view : str
        The name of the view to plot the loadings for.
    factor : int, optional
        The index of the factor to plot the loadings for. Default is 0.
    Returns
    -------
    matplotlib Axes object
         Plot with loadings for the specified factor and view.
    """
    
    if not hasattr(model, f"W"):
        model.W = [model.predict(f"W_{i}", num_split=10000) for i in range(len(model.X))] 
    W = get_W(model, view)
    W = W.loc[factor, :]
    W_sorted = W.sort_values()
    labels = np.array(W_sorted.index.tolist())
    if len(labels) > 20:
        labels[10:len(labels)-10] = ""
    x = [i for i in range(len(W_sorted))]
    fig, ax = plt.subplots(1)
    ax.scatter(x, W_sorted)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], W_sorted[i]))
    return ax


def plot_top_loadings(
        model: SOFA,
        view: int,
        factor: int,
        top_n:  int=10,
        sign: Union[None,str]=None, 
        highlight: Union[None,str]=None,
        ax: Union[None,Axes]=None) -> Axes:
    """
    Plot the top loadings of a factor in a given view.

    Parameters
    ----------
    model : SOFA
        The trained model object.
    view : int
        The index of the view to plot.
    factor : str
        The name of the factor to plot.
    top_n : int, optional
        The number of top loadings to plot, by default 10.
    sign : str, optional
        The sign of the loadings to plot. If None, plot the top absolute loadings.
        If "+" or "-", plot the top positive or negative loadings, respectively.
    highlight : list, optional
        A list of feature names to highlight in red, by default None.
    ax : matplotlib Axes object, optional
        The axes to plot on. If None, a new figure is created.
    Returns
    -------
    matplotlib  Axes object
        Plot with top loadings.
    """
    if not hasattr(model, f"W"):
        model.W = [model.predict(f"W_{i}", num_split=10000) for i in range(len(model.X))]
        
    W = get_W(model, view)
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
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.hlines(y=my_range, xmin=0, xmax=topW,alpha=0.4)
    ax.scatter(topW, my_range, alpha=1)
     
    # Add title and axis names
    ax.set_yticks(my_range, topW.index)
    signoffset = max(topW)/50
    if sign == "-":
        for i in range(len(topW)):
            ax.annotate("-", (topW[i]+signoffset*1.4,i+0.65), fontsize=16)
    if sign == "+":
        for i in range(len(topW)):
            ax.annotate("+", (topW[i]+signoffset,i+0.7), fontsize=12)
    ax.set_xlabel('Loadings')
    ax.set_ylabel('Features')
    if highlight != None:
        for j in highlight_id:
            plt.setp(ax.get_yticklabels()[j], color='red')

    return ax

def plot_variance_explained(
        model: SOFA,
        ax: Union[None,Axes]=None
        ) -> Axes:
    """
    Plot the variance explained by each factor for each view.

    Parameters
    ----------
    model : SOFA
        The trained SOFA model.
    ax : matplotlib Axes object, optional
        The axes to plot on. If None, a new figure is created.
    Returns
    -------
    matplotlib Axes object
        Plot with variance explained by each factor for each view.
    """
    
    vexp = get_var_explained_per_view_factor(model)
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

    if ax is None:
        fig, ax = plt.subplots()
    plot = ax.imshow(vexp, cmap="Blues", origin="lower")

    ax.set_xlabel("View")
    ax.set_xticks(ticks = range(len(model.views)), labels= model.views,rotation=90)
    ax.set_yticks(ticks = range(vexp.shape[0]), labels= y_labels)
    ax.set_ylabel("Factor")
    plt.colorbar(plot)
    return ax

def plot_variance_explained_factor(model: SOFA,
                                   ax: Union[None,Axes]=None
                                   ) -> Axes:

    """
    Plots the variance explained by each factor.

    Parameters
    ----------
    model : SOFA
        The trained SOFA model.
    ax : matplotlib Axes object, optional
        The axes to plot on. If None, a new figure is created.
    Returns
    -------
    matplotlib Axes object
        Plot with variance explained by each factor.
    """    
 
    vexp = get_var_explained_per_view_factor(model)
    vexp = np.sum(vexp, axis=1)
    if model.Ymdata is not None:
        x_labels = np.array(["" for i in range(model.num_factors)], dtype=object)
        guided_factors = list(model.Ymdata.mod.keys())
        for i in range(len(guided_factors)):
            s =  "(" + guided_factors[i] + ")"
            x_labels[model.design.cpu().numpy()[i,:]==1] =s

        for i in range(model.num_factors):
            x_labels[i] = x_labels[i] + f" {i+1}"
    else:
        x_labels = np.arange(1,model.num_factors+1)
    fig, ax = plt.subplots(1)
    plot = ax.bar(["Factor"+ str(i+1) for  i in range(len(vexp))], vexp)
    ax.set_xlabel("View")
    ax.set_ylabel("R2")
    ax.set_xticklabels(x_labels,rotation=90)
    return ax

def plot_variance_explained_view(model: SOFA,
                                 ax: Union[None,Axes]=None
                                ) -> Axes:
    """
    Plots the variance explained of each view.

    Parameters
    ----------
    model : SOFA
        The trained SOFA model.
    ax : matplotlib Axes object, optional
        The axes to plot on. If None, a new figure is created.
    Returns
    -------
    Axes
        Plot with variance explained of each view.
    """    
    X = [i.cpu().numpy() for i in model.X]

    vexp = []
    if not hasattr(model, f"X_pred"):
        model.X_pred = [model.predict(f"X_{i}", num_split=10000) for i in range(len(X))]
        
    for i in range(len(X)):
        mask = model.Xmask[i].cpu().numpy()
        vexp.append(calc_var_explained(model.X_pred[i][mask,:], X[i][mask,:]))

    fig, ax = plt.subplots(1)
    plot = ax.bar(["view"+ str(i) for  i in range(len(vexp))], vexp)
    ax.set_xlabel("View")
    ax.set_ylabel("R2")
    ax.set_xticks(ticks = range(len(model.views)), labels= model.views,rotation=90)
    return ax

def plot_factor_covariate_cor(
        model: SOFA,
        metavar: List[int],
        ax: Union[None,Axes]=None
        ) -> Axes:
    """
    Plot the correlation between the factors and covariates.

    Parameters
    ----------
    model : SOFA
        The trained SOFA model.
    metavar : list of str
        The list of covariate names to plot
    ax : matplotlib Axes object, optional
        The axes to plot on. If None, a new figure is created.
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
        y_labels = model.Ymdata.mod
        y_labels = ["(" +i +") " for i in y_labels]
        y_labels = y_labels + ["" for i in range(model.Z.shape[1] - len(y_labels))]
        y_labels = [i+ str(idx+1) for idx,i in enumerate(y_labels)]
    else:
        y_labels = np.arange(1,model.num_factors+1)
    cormat = np.stack(cor).T

    divnorm=colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1)
    
    if ax is None:
        fig, ax = plt.subplots()
    plot = ax.imshow(cormat, cmap="RdBu", origin="lower", norm=divnorm)
    ax.set_xlabel("Covariate")
    ax.set_xticks(ticks = range(len(metavar)), labels= metavar, rotation=90)
    ax.set_yticks(ticks = range(cormat.shape[0]), labels= y_labels)
    ax.set_ylabel("Factor")
    plt.colorbar(plot)

    return ax



def plot_fit(
        model: SOFA,
        view: str
        ) -> Axes:
    """
    Plot the scatter plot with predicted X vs input X, to assess the model fit.

    Parameters
    ----------
    model : SOFA
        The trained SOFA model.
    view : str
        The name of the view to plot.

    Returns
    -------
    matplotlib Axes object
        Scatter plot with predicted X vs input X.
    """
    X = model.X[model.views.index(view)].cpu().numpy()
    X_pred = model.X_pred[model.views.index(view)]

    fig, ax = plt.subplots(1)
    ax.scatter(X, X_pred, alpha=0.2, s=1)
    ax.plot([np.min(X_pred), np.max(X_pred)], [np.min(X_pred), np.max(X_pred)], color = "black")
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("X predicted")
    return ax

def _abs_formatter(x, pos):
    return f"{abs(x):.0f}"

def plot_enrichment(
    gene_list: list, 
    background: list,
    db: list,
    top_n: list,
    ax: Union[None,Axes]=None,
    )-> Axes:
    """
    Plot bar plot of adjusted p-values for gene set overrepresentation analysis based on a provided gene_list. 
    The gene set overrepresentation analysis uses the enrichr api and calculates whether  
    gene sets from chosen databases are significantly overrepresented vs a background gene set.

    Parameters
    ----------
    gene_list : list
        Gene list to perform gene set overrepresentation analysis on.
    background : list
        Background gene list, a good choice would be all the genes that are considered in the analysis 
        (i.e. all the genes in X).
    db : list
        List of strings of databases to perform gene set overrepresentation analysis on. A list of possible 
        databases can be found here https://maayanlab.cloud/Enrichr/#libraries.
    top_n : list
        List of integers of the number of top gene sets to plot for each database.
    ax : Union[None,Axes], optional
        Axes to plot on, if None, a new figure is created, by default None.

    Returns
    -------
    matplotlib Axes object
        Bar plot of adjusted p-values for gene set overrepresentation analysis.
    """

    cblind_colors = {db[i]:sns.color_palette("colorblind", as_cmap=True)[i] for i in range(len(db))}

    categories = []
    values = []
    dbs = []
    for i in range(len(db)):
        enr = get_gsea_enrichment(gene_list, db=db[i], background=background)
        res = enr.results.head(top_n[i])
        categories_ = list(reversed(res["Term"].tolist()))
        categories.append(categories_)
        values.append(list(reversed(-np.log10(res["Adjusted P-value"]))))
        dbs.append(db[i])
        
    if ax is None:
        fig, ax = plt.subplots(1)
    # Create horizontal bar plot
    for i in range(len(categories)):
        ax.barh(categories[i], values[i], label=dbs[i], color = cblind_colors[dbs[i]])
    
    # Add labels and title
    ax.set_xlabel('-log10 adjusted p-values')
    ax.set_ylabel('Terms')

    plt.gca().xaxis.set_major_formatter(FuncFormatter(_abs_formatter))
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),loc='upper center', bbox_to_anchor=(0, -0.2), ncol=int(len(db)/2+1))
    return ax