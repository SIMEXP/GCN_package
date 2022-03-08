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
FEATURE_SIZE = 5
OUT_FEATURE_SIZE = FEATURE_SIZE - 2

def test_chebnet():
    # create toy graph compatible with torch_geometric, and input
    adjacency_matrix = np.random.rand(NUM_NODES, NUM_NODES).astype(np.float32)
    adjacency_matrix = adjacency_matrix*(adjacency_matrix>0.8)
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.transpose())/2
    np.fill_diagonal(adjacency_matrix, 1)
    adj_sparse = tg.utils.dense_to_sparse(torch.from_numpy(adjacency_matrix))
    graph = tg.data.Data(edge_index=adj_sparse[0], edge_attr=adj_sparse[1])
    # one batch with 4 nodes that has a feature array of size 10
    x = torch.from_numpy(np.random.randn(NUM_NODES, FEATURE_SIZE).astype(np.float32))
    # model
    model = gcn_package.models.gcn.ChebConvBlock(graph.edge_index, graph.edge_attr, gcn_filter_size=2,
                                                in_filters=FEATURE_SIZE, OUT_filters=OUT_FEATURE_SIZE, gcn_normalization="sym", bias=True, activation="ReLU", dropout=0.2)
    out = model(x)
                                 
    assert out.shape == (NUM_NODES, OUT_FEATURE_SIZE)