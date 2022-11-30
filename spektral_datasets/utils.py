import numpy as np
from sklearn.neighbors import kneighbors_graph


def normalize_xy(x, y):
    """
    Linearly scale x and y to between 0 and 1 using min/max values (ignoring nan values)
    """
    x = (x - np.nanmin(x, axis=0, keepdims=True)) / (
            np.nanmax(x, axis=0, keepdims=True) - np.nanmin(x, axis=0, keepdims=True))
    y = (y - np.nanmin(y, axis=0, keepdims=True)) / (
            np.nanmax(y, axis=0, keepdims=True) - np.nanmin(y, axis=0, keepdims=True))
    return x, y


def get_adj(x, k):
    print("Computing k neighbors graph...")
    a = kneighbors_graph(x, k, include_self=False)
    a = a + a.T  # to make graph symmetric (using k neighbours in "either" rather than "mutual" mode)
    a[a > 1] = 1  # get rid of any edges we just made double
    print("Graph computed.")
    return a
