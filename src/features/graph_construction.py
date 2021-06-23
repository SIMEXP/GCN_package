import numpy as np
import networkx as nx
import torch_geometric as tg

def make_undirected(mat):
    """Takes an input adjacency matrix and makes it undirected (symmetric).

    Parameters
    ----------
    mat: array
        Square adjacency matrix.

    Raises
    ------
    ValueError
        If input matrix is not square.

    Returns
    -------
    array
        Symmetric input matrix. If input matrix was unweighted, output is also unweighted.
        Otherwise, output matrix is average of corresponding connection strengths of input matrix.
    """
    if not (mat.shape[0] == mat.shape[1]):
        raise ValueError('Adjacency matrix must be square.')

    sym = (mat + mat.transpose())/2
    if len(np.unique(mat)) == 2: #if graph was unweighted, return unweighted
        return np.ceil(sym) #otherwise return average
    return sym

# ANNABELLE
def knn_graph(mat,k=8,selfloops=False,symmetric=True):
    """Takes an input matrix and returns a k-Nearest Neighbour weighted adjacency matrix.

    Parameters
    ----------
    mat: array
        Input adjacency matrix, can be symmetric or not.
    k: int, default=8
        Number of neighbours.
    selfloops: bool, default=False
        Wether or not to keep selfloops in graph, if set to False resulting adjacency matrix
        has zero along diagonal.
    symmetric: bool, default=True
        Wether or not to return a symmetric adjacency matrix. In cases where a node is in the neighbourhood
        of another node that is not its neighbour, the connection strength between the two will be halved.
    
    Raises
    ------
    ValueError
        If input matrix is not square.
    
    Returns
    -------
    array
        Adjacency matrix of k-Nearest Neighbour graph.
    """
    if not (mat.shape[0] == mat.shape[1]):
        raise ValueError('Adjacency matrix must be square.')

    dim = mat.shape[0]
    if (k<=0) or (dim <=k):
        raise ValueError('k must be in range [1,n_nodes)')

    m = np.abs(mat) # Look at connection strength only, not direction
    mask = np.zeros((dim,dim),dtype=bool)
    for i in range(dim):
        sorted_ind = m[:,i].argsort().tolist()
        del sorted_ind[sorted_ind.index(i)] # Remove self from neighbour list
        neighbours = sorted_ind[-k:]
        mask[:,i][neighbours] = True
    adj = mat.copy() # Original connection strengths
    adj[~mask] = 0

    if not selfloops:
        np.fill_diagonal(adj,0)

    if symmetric:
        return make_undirected(adj)
    else:
        return adj

# LOIC
def knn_graph_quantile(corr_matrix, k=8):
    """Takes an input correlation matrix and returns a k-Nearest Neighbour weighted undirected adjacency matrix."""
    is_undirected = (corr_matrix == corr_matrix.T).all()
    m = corr_matrix.copy()
    # absolute correlation
    m = np.abs(m)
    np.fill_diagonal(m, 0)
    # knn graph with quantile
    quantile_k = np.quantile(m, k/m.shape[0], axis=-1)
    mask_not_neighbours = (m > quantile_k)
    m[mask_not_neighbours] = 0
    if is_undirected:
        return m
    else:
        return make_undirected(m)
    
def make_group_graph(connectomes,k=8):
    # Group average connectome
    avg_conn = np.array(connectomes).mean(axis=0)

    # Undirected 8 k-NN graph as matrix
    avg_conn8 = knn_graph(avg_conn,k=k)

    # Format matrix into graph for torch_geometric
    graph = nx.convert_matrix.from_numpy_array(avg_conn8)
    return tg.utils.from_networkx(graph)