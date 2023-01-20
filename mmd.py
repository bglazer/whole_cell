#%%     
# Taken from: 
# https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py

import torch
#%%
def rbf(X, Y, gamma):
    lx = X.shape[0]
    ly = Y.shape[0]
    # Get all pairs of (x, y) in X and Y
    idxs = torch.cartesian_prod(torch.arange(lx), torch.arange(ly))
    rX = X[idxs[:,0]]
    rY = Y[idxs[:,1]]
    
    return torch.exp(-gamma * torch.sum(rX-rY, dim=1)**2)

def mmd(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = rbf(X, X, gamma)
    YY = rbf(Y, Y, gamma)
    XY = rbf(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()  
# %%
