import numpy as np
import networkx as nx
import torch_geometric as tg

def make_undirected(mat):
    """Takes an input adjacency matrix and makes it undirected (symmetric)."""
    if not (mat.shape[0] == mat.shape[1]):
        raise ValueError('Adjacency matrix must be square.')
        
    sym = (mat + mat.transpose())/2
    if len(np.unique(mat)) == 2: #if graph was unweighted, return unweighted
        return np.ceil(sym) #otherwise return average
    return sym

# ANNABELLE
#def knn_graph(mat,k=8):
#    """Takes an input matrix and returns a k-Nearest Neighbour weighted adjacency matrix."""
#    is_undirected = (mat == mat.T).all()
#    m = np.abs(mat.copy())
#    np.fill_diagonal(m,0)
#    slices = []
#    for i in range(m.shape[0]):
#        s = m[:,i]
#        not_neighbours = s.argsort()[:-k]
#        s[not_neighbours] = 0
#        slices.append(s)
#    if is_undirected:
#        return np.array(slices)
#    else:
#        return make_undirected(np.array(slices))
    
def make_group_graph(connectomes,k=8):
    # Group average connectome
    avg_conn = np.array(connectomes).mean(axis=0)

    # Undirected 8 k-NN graph as matrix
    avg_conn8 = knn_graph(avg_conn,k=k)

    # Format matrix into graph for torch_geometric
    graph = nx.convert_matrix.from_numpy_array(avg_conn8)
    return tg.utils.from_networkx(graph)

# LOIC
def knn_graph(corr_matrix, k=8):
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
