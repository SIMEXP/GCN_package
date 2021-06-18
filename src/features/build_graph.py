import os
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch_geometric as tg
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split

data_path = os.path.join("..", "..", "data", "cobre_difumo512")
ts_path = os.path.join(data_path, "difumo", "timeseries")
conn_path = os.path.join(data_path, "difumo", "connectomes")
pheno_path = os.path.join(data_path, "difumo", "phenotypic_data.tsv")

timeseries = [np.load(os.path.join(ts_path, p)) for p in os.listdir(ts_path)]
ids = [int(p.split('_')[1]) for p in os.listdir(ts_path)]

# One subject has different length timeseries, ignore them for now
not_150 = np.array([t.shape[0]!=150 for t in timeseries])
print('Bad sub ID: {}'.format(np.array(ids)[not_150][0]))

def make_undirected(mat):
    """Takes an input adjacency matrix and makes it undirected (symmetric)."""
    m = mat.copy()
    mask = mat != mat.transpose()
    vals = mat[mask] + mat.transpose()[mask]
    m[mask] = vals
    return m

def knn_graph(mat, k=8, directed=False):
    """Takes an input matrix and returns a k-Nearest Neighbour weighted adjacency matrix."""
    m = mat.copy()
    np.fill_diagonal(m,0)
    slices = []
    for i in range(m.shape[0]):
        s = m[:,i]
        not_neighbours = s.argsort()[:-k]
        s[not_neighbours] = 0
        slices.append(s)

    is_unidrected = (mat == mat.T).all()
    m2 = mat.copy()
    # absolute correlation
    m2 = np.abs(m2)
    np.fill_diagonal(m2, 0)
    # knn graph with quantile
    quantile_k = np.quantile(m2, k/m2.shape[0], axis=-1)
    mask_not_neighbours = (m2 > quantile_k)
    m2[mask_not_neighbours] = 0


    if not directed:
        return np.array(slices)
    else:
        return make_undirected(np.array(slices))

def new_knn_graph(corr_matrix, k=8):
    """Takes an input correlation matrix and returns a k-Nearest Neighbour weighted unidrected adjacency matrix."""
    is_unidrected = (corr_matrix == corr_matrix.T).all()
    m = corr_matrix.copy()
    # absolute correlation
    m = np.abs(m)
    np.fill_diagonal(m, 0)
    # knn graph with quantile
    quantile_k = np.quantile(m, k/m.shape[0], axis=-1)
    mask_not_neighbours = (m > quantile_k)
    m[mask_not_neighbours] = 0
    if is_unidrected:
        return m
    else:
        return make_undirected(m)
    
def make_group_graph(conn_path):
    # Load connectomes
    connectomes = [np.load(os.path.join(conn_path,p)) for p in os.listdir(conn_path)]

    # Group average connectome
    avg_conn = np.array(connectomes).mean(axis=0)

    # Undirected 8 k-NN graph as matrix
    avg_conn8 = knn_graph(avg_conn, directed=False)
    # Undirected 8 k-NN graph as matrix
    new_avg_conn8 = new_knn_graph(avg_conn)

    # Format matrix into graph for torch_geometric
    graph = nx.convert_matrix.from_numpy_array(avg_conn8)
    return tg.utils.from_networkx(graph)

make_group_graph(conn_path)