
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
def plot_weights(W , factor =0): 
    W_sorted = W.loc[factor,:].sort_values()
    labels = np.array(W_sorted.index.tolist())
    labels[10:len(labels)-10] = ""
    x = [i for i in range(len(W_sorted))]
    fig, ax= plt.subplots(1)
    ax.scatter(x, W_sorted)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], W_sorted[i]))
        
    return 
        
def plot_top_weights(W, factor = 0, top_n = 10, sign=None, highlight=None):
    
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

def plot_variance_explained(model, X):

    vexp = []
    for i in range(len(X)):
        X_pred = model.get_Xpred_perfactor(view=i)
        vexp.append(calc_var_explained(X_pred, X[i].numpy()).reshape(model.num_factors,1))

    vexp = np.hstack(vexp)
    
    fig, ax = plt.subplots(1)
    plot = plt.imshow(vexp, cmap="Blues", origin="lower")
    plt.xlabel("View")
    plt.xticks(ticks = range(len(model.views)), labels= model.views)
    plt.yticks(ticks = range(vexp.shape[0]), labels= range(1,vexp.shape[0]+1))
    plt.ylabel("Factor")
    plt.colorbar(plot)
    return plot, vexp

def plot_variance_explained_view(model, X):
    


    vexp = []
    for i in range(len(X)):
        X_pred = model.get_Xpred(view=i)
        vexp.append(calc_var_explained_view(X_pred, X[i].numpy()))

    fig, ax = plt.subplots(1)
    plot = plt.bar(["view"+ str(i) for  i in range(len(vexp))], vexp)
    plt.xlabel("View")
    plt.ylabel("R2")
    plt.xticks(ticks = range(len(model.views)), labels= model.views)
    return plot, vexp