import pyro
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import scipy.stats as stats

def plot_weights(model, view, factor =0):
    if not hasattr(model, f"W"):
        model.W = [model.predict(f"W_{i}", num_split=10000) for i in range(len(X))]
        
    W = pd.DataFrame(model.W[view], columns = model.Xmdata.mod[list(model.Xmdata.mod.keys())[view]].var_names)
    
    W = W.loc[factor,:]
    W_sorted = W.sort_values()
    labels = np.array(W_sorted.index.tolist())
    labels[10:len(labels)-10] = ""
    x = [i for i in range(len(W_sorted))]
    fig, ax= plt.subplots(1)
    ax.scatter(x, W_sorted)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], W_sorted[i]))
    plt.show()
    plt.close()
    return 
        
def plot_top_weights(model, view, factor, top_n = 10, sign=None, highlight=None):
    
    if not hasattr(model, f"W"):
        model.W = [model.predict(f"W_{i}", num_split=10000) for i in range(len(X))]
        
    W = pd.DataFrame(model.W[view], columns = model.Xmdata.mod[list(model.Xmdata.mod.keys())[view]].var_names)
    
    W = W.loc[factor,:]

    if sign ==None:
        idx = np.argpartition(abs(W), -top_n)[-top_n:]
        #order = W.map(lambda x : x).sort_values(ascending = False)
        # topW = W[order.index][0:top_n]
        # topW = topW.iloc[::-1]
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
    plt.hlines(y=my_range, xmin=0, xmax=topW,alpha=0.4)
    plt.scatter(topW, my_range, alpha=1)
     
    # Add title and axis names
    plt.yticks(my_range, topW.index)
    signoffset = max(topW)/50
    if sign == "-":
        for i in range(len(topW)):
            plt.annotate("-", (topW[i]+signoffset,i+0.7), fontsize=16)
    if sign == "+":
        for i in range(len(topW)):
            plt.annotate("+", (topW[i]+signoffset,i+0.7), fontsize=12)
    plt.xlabel('Loadings')
    plt.ylabel('Features')
    if highlight != None:
        for j in highlight_id:
            plt.setp(ax.get_yticklabels()[j], color='red')
    plt.tight_layout()
    plt.show()
    plt.close()
    return 


def calc_var_explained(X_pred, X):
    vexp = []

    for i in range(len(X_pred)):
        num = np.sum(np.square(X-X_pred[i]))
        denom = np.sum(np.square(X))
        vexp.append(1 - num/denom)
    vexp = np.stack(vexp)
    vexp[vexp < 0] = 0
    return vexp

def calc_var_explained_view(X_pred, X):
    vexp = []
    

    vexp = 1 - np.sum((np.square(X-X_pred)))/np.sum(np.square(X))
    if vexp < 0:
        vexp = 0
    return vexp



def plot_variance_explained(model):
    #params = {i: j for i, j in pyro.get_param_store().items()}
    X = [i.cpu().numpy() for i in model.X]
    vexp = []
    if not hasattr(model, "Z"):
        model.Z = model.predict("Z", num_split=10000)
    if not hasattr(model, f"W"):
        model.W = [model.predict(f"W_{i}", num_split=10000) for i in range(len(X))]   
    for i in range(len(X)):
        mask = model.Xmask[i].cpu().numpy()
        X_pred_factor = []
        #offset = params[f"mean_{i}"].detach().cpu().numpy()

        for j in range(model.num_factors):
            X_pred_factor.append(model.Z[mask,j, np.newaxis] @ model.W[i][np.newaxis,j,:])
        vexp.append(calc_var_explained(X_pred_factor, X[i][mask,:]).reshape(model.num_factors,1))
    vexp = np.hstack(vexp)
    if model.Ymdata is not None:

        y_labels = model.Ymdata.mod
        y_labels = ["(" +i +") " for i in y_labels]
        y_labels = y_labels + ["" for i in range(model.Z.shape[1] - len(y_labels))]
        y_labels = [i+ str(idx+1) for idx,i in enumerate(y_labels)]
    else:
        y_labels = np.arange(1,model.num_factors+1)
    fig, ax = plt.subplots(1)
    plot = plt.imshow(vexp, cmap="Blues", origin="lower")
    plt.xlabel("View")
    plt.xticks(ticks = range(len(model.views)), labels= model.views,rotation=90)
    plt.yticks(ticks = range(vexp.shape[0]), labels= y_labels)
    plt.ylabel("Factor")
    #labels = plt.gca().get_yticklabels()
    #label_to_change = labels[6]
    #label_to_change. set_color('red')
    plt.colorbar(plot)
    plt.tight_layout()
    return plot, vexp

def plot_variance_explained_factor(model):
    #params = {i: j for i, j in pyro.get_param_store().items()}
    X = [i.cpu().numpy() for i in model.X]
    vexp = []
    if not hasattr(model, "Z"):
        model.Z = model.predict("Z", num_split=10000)
    if not hasattr(model, f"W"):
        model.W = [model.predict(f"W_{i}", num_split=10000) for i in range(len(X))]   
    for i in range(len(X)):
        mask = model.Xmask[i].cpu().numpy()
        X_pred_factor = []
        #offset = params[f"mean_{i}"].detach().cpu().numpy()

        for j in range(model.num_factors):
            X_pred_factor.append(model.Z[mask,j, np.newaxis] @ model.W[i][np.newaxis,j,:])
        vexp.append(calc_var_explained(X_pred_factor, X[i][mask,:]).reshape(model.num_factors,1))
    vexp = np.hstack(vexp)
    vexp = np.sum(vexp, axis=1)
    fig, ax = plt.subplots(1)
    plot = plt.bar(["Factor"+ str(i+1) for  i in range(len(vexp))], vexp)
    plt.xlabel("View")
    plt.ylabel("R2")
    plt.xticks(ticks = range(len(vexp)),rotation=90)
    plt.show()
    plt.close()
    return plot, vexp

def plot_variance_explained_view(model):
    

    X = [i.cpu().numpy() for i in model.X]

    vexp = []
    if not hasattr(model, f"X_pred"):
        model.X_pred = [model.predict(f"X_{i}", num_split=10000) for i in range(len(X))]
        
    for i in range(len(X)):
        mask = model.Xmask[i].cpu().numpy()
        vexp.append(calc_var_explained_view(model.X_pred[i][mask,:], X[i][mask,:]))

    fig, ax = plt.subplots(1)
    plot = plt.bar(["view"+ str(i) for  i in range(len(vexp))], vexp)
    plt.xlabel("View")
    plt.ylabel("R2")
    plt.xticks(ticks = range(len(model.views)), labels= model.views,rotation=90)
    plt.show()
    plt.close()
    return plot, vexp

def plot_factor_covariate_cor(model, metavar=None): 
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
    #for i in Ymdata.mod:
    #    if Ymdata.mod[i].X.dtype.type is np.string_:
    #        lmap = {l:j for j,l in enumerate(np.unique(Ymdata.mod[i].X))}
    #        y = np.array([lmap[l] for l in Ymdata.mod[i].X])
    #    else:
    #        y = Ymdata.mod[i].X.flatten()
    #    cor.append([stats.pearsonr(Z_pred[:,z], y)[0] for z in range(Z_pred.shape[1])])
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
    plot = plt.imshow(abs(cormat), cmap="Oranges", origin="lower")
    plt.xlabel("Covariate")
    #plt.xticks(ticks = range(len(model.Ymdata.mod)), labels= model.Ymdata.mod, rotation=90)
    plt.xticks(ticks = range(len(metavar)), labels= metavar, rotation=90)
    plt.yticks(ticks = range(cormat.shape[0]), labels= y_labels)
    plt.ylabel("Factor")
    labels = plt.gca().get_yticklabels()
    #label_to_change = labels[6]
    #label_to_change. set_color('red')
    plt.colorbar(plot)
    plt.tight_layout()


def plot_fit(model, view):
    # check reconstructions of data
    X = model.X[view].cpu().numpy()
    X_pred = model.X_pred[view]
    fig, ax = plt.subplots(1)
    ax.scatter(X, X_pred, alpha=0.2, s=1)
    ax.plot([np.min(X_pred), np.max(X_pred)], [np.min(X_pred), np.max(X_pred)], color = "black")
    ax.set_aspect('equal')
    plt.xlabel("X")
    plt.ylabel("X predicted")
    plt.show()
    plt.close()
