import os
import sys
import numpy as np
import torch
import torch_geometric as tg
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import gcn_package
import gcn_package.visualization.visualize
import gcn_package.models.gcn
import gcn_package.features.graph_construction
import gcn_package.data.time_windows_dataset
import gcn_package.data.raw_data_loader

# global parameters
random_seed = 0
np.random.seed(random_seed)
NUM_NODES = 10
FEATURE_SIZE = 40
OUT_FEATURE_SIZE = FEATURE_SIZE - 2
BATCH_SIZE = 3

def test_chebconvblock():
    # create toy graph compatible with torch_geometric, and input
    adjacency_matrix = np.random.rand(NUM_NODES, NUM_NODES).astype(np.float32)
    adjacency_matrix = adjacency_matrix*(adjacency_matrix>0.8)
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.transpose())/2
    np.fill_diagonal(adjacency_matrix, 1)
    adj_sparse = tg.utils.dense_to_sparse(torch.from_numpy(adjacency_matrix))
    graph = tg.data.Data(edge_index=adj_sparse[0], edge_attr=adj_sparse[1])
    # one batch with 4 nodes that has a feature array of size 10
    x = torch.from_numpy(np.random.randn(BATCH_SIZE, NUM_NODES, FEATURE_SIZE).astype(np.float32))
    # model
    model = gcn_package.models.gcn.ChebConvBlock(graph.edge_index, graph.edge_attr, gcn_filter_size=2,
                                                in_filters=FEATURE_SIZE, out_filters=OUT_FEATURE_SIZE, gcn_normalization="sym", bias=True, activation="ReLU", dropout=0.2)
    out = model(x)
    print("ChebConvBlock output: {}".format(out.shape))
    assert out.shape == (BATCH_SIZE, NUM_NODES, OUT_FEATURE_SIZE)

def test_linearblock():
    x = torch.from_numpy(np.random.randn(BATCH_SIZE, FEATURE_SIZE).astype(np.float32))
    model = gcn_package.models.gcn.LinearBlock(in_filters=FEATURE_SIZE, out_filters=OUT_FEATURE_SIZE, activation="ReLU", batch_normalisation=True, dropout=0.2)
    out = model(x)
    print("LinearBlock output: {}".format(out.shape))
    assert out.shape == (BATCH_SIZE, OUT_FEATURE_SIZE)

def test_customgcn():
    # create toy graph compatible with torch_geometric, and input
    adjacency_matrix = np.random.rand(NUM_NODES, NUM_NODES).astype(np.float32)
    adjacency_matrix = adjacency_matrix*(adjacency_matrix>0.8)
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.transpose())/2
    np.fill_diagonal(adjacency_matrix, 1)
    adj_sparse = tg.utils.dense_to_sparse(torch.from_numpy(adjacency_matrix))
    graph = tg.data.Data(edge_index=adj_sparse[0], edge_attr=adj_sparse[1])
    # one batch with 4 nodes that has a feature array of size 10
    x = torch.from_numpy(np.random.randn(BATCH_SIZE, NUM_NODES, FEATURE_SIZE).astype(np.float32))
    model = gcn_package.models.gcn.CustomGCN(edge_index=graph.edge_index, edge_weight=graph.edge_attr, n_timepoints=FEATURE_SIZE, n_roi=NUM_NODES, n_classes=2,
                 n_gcn_layers=6, gcn_filters=32, gcn_filter_size=2, gcn_filter_decreasing_rate=1, gcn_normalization="sym",
                 n_linear_layers=3, linear_filters=16, linear_filter_decreasing_rate=2, batch_normalisation=True,
                 bias=True, activation="ReLU", dropout=0.2)
    out = model(x)

if __name__ == "__main__":
    test_chebconvblock()
    test_linearblock()
    test_customgcn()