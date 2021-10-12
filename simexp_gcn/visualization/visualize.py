import os
import torch
import copy
import sklearn
import matplotlib.pyplot as plt
import torch_geometric as tg
import nilearn as nil
import nilearn.input_data
import simexp_gcn
import simexp_gcn.data.raw_data_loader
import simexp_gcn.features.graph_construction

# TODO provide visualization tools
# 1. Basic graph plotting using https://networkx.org/documentation/stable/tutorial.html
# 2. Eigen vector representation: https://persagen.com/files/misc/arxiv-1712.00468.png
# 3. Basic clustering: https://github.com/neurolibre/gcn_tutorial_test
# 4. GCN latent space representation (T-SNE or PCA)
# 5. Graph signal reconstruction (for graph-AE)


def PCA(data):
    """Compute PCA using scikit-learn

    Parameters
    ----------
    data: `numpy array` of shape (n_samples, n_features)
      Training data

    Returns
    -------
    out: `numpy array`
      Transformed values.
    """
    # normalization and pca decomposition
    normalized_data = sklearn.preprocessing.scale(data)
    PCA = sklearn.decomposition.PCA(n_components=2)
    pca_results = PCA.fit_transform(normalized_data)

    return pca_results


class GetActivation():
    def __init__(self, layer_name=None):
        self.layer_name = layer_name
        self.outputs = {}

    def __call__(self, module, module_in, module_out):
        self.outputs[self.layer_name] = module_out.detach()

    def clear(self):
        self.outputs = {}


def visualize_activation(model, data_generator, layer_name):
    """Visualize the activation layer from a PyTorch trained model

    Parameters
    ----------
    model: `torch.nn.Module`
      Trained model.
    data_generator: `torch.utils.data.dataloader.DataLoader`
      The pytorch data generator to use.
    layer_name: `str`
      Layer name to compute activation.
    """
    # copying input model to not alter it
    gcn_test = copy.deepcopy(model)
    # adding forward hook for the specified layer
    get_activation = GetActivation(layer_name)
    for curr_layer_name, layer in gcn_test.named_modules():
        if curr_layer_name == layer_name:
            layer.register_forward_hook(get_activation)
    # evaluation of the model with the data_generator
    activations = []
    labels = []
    gcn_test.eval()
    for x, y in data_generator:
        labels += [y]
        out = gcn_test.forward(x)
        activations += [get_activation.outputs[layer_name]]
    activations = torch.vstack(activations)
    labels = torch.hstack(labels)
    pca_results = PCA(activations)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    for curr_label in torch.unique(labels):
        mask = labels == curr_label
        axes.scatter(
            x=pca_results[mask, 0], y=pca_results[mask, 1], label=f"class {curr_label}")
    axes.legend()
    axes.set_title("PCA decomposition of {} activation".format(layer_name))
    plt.show()


def test():
    a = 0
    return a


def vizualize_weights(model, layer_name):
    """ Visualize the weights from a specific layer of a PyTorch trained model
    Parameters
    ----------
    model: `torch.nn.Module`
      Trained model.
    layer_name: `str`
      Layer name to visualize weights.
    """
    last_layer_parameters = model.state_dict()[layer_name + ".weight"].numpy()
    # sample (weights to visualize) should be in x axis for sklearn
    last_layer_parameters = last_layer_parameters.T
    pca_results = PCA(last_layer_parameters)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.scatter(x=pca_results[:, 0], y=pca_results[:, 1])
    axes.set_title("PCA decomposition of {} weights".format(layer_name))

    plt.show()


def embedding_error(func_img, graph, maps_img, confounds=None):
    """ Plot and compute the reconstruction error of fMRI data into spectral domain

    Parameters
    ----------
    func_img: 4D `nibabel.nifti1.Nifti1Image`
      Input data to embed and reconstruct.
    graph: `torch_geometric.data`
      Usually a connectome, or diffusion matrix in PyTorch format.
    maps_img: 4D `nibabel.nifti1.Nifti1Image`
      Set of continuous maps for nilearn masker.
    confounds: `string` or 2D `np.array`
      CSV file or array-like representing the signal(s) to filter out.
    """

    # extract timeserie data
    masker = nil.input_data.NiftiMapsMasker(
        maps_img=maps_img, standardize=True, verbose=2, memory_level=0)
    data = masker.fit_transform(func_img, confounds=confounds)
    # compute the laplacian matrix: L=I-D(-1/2)AD(-1/2)
    degree = tg.utils.degree(graph.edge_index[0, :])
    degree = torch.diag(degree).pow(-0.5)
    degree[torch.isinf(degree)] = 0
    adj = tg.utils.to_dense_adj(graph.edge_index)[0, ...]
    id = torch.diag(torch.ones(adj.shape[0]))
    lap = id - torch.matmul(torch.matmul(degree, adj), degree)
    print("DONE")

# t0 = time.time()
# adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat_sp)
# print("Converting from scipy sparse matrix:")
# print(adj_mat_sp.indices)
# print(adj_mat_sp.data)
# print("Converting to torch sparse tensor:")
# print(adj_mat._indices())
# print(adj_mat._values())

# edge_index = adj_mat._indices()
# edge_weight = adj_mat._values()
# row, col = edge_index

# #degree-matrix
# deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

# # Compute normalized and rescaled Laplacian.
# deg = deg.pow(-0.5)
# deg[torch.isinf(deg)] = 0
# lap = deg[row] * edge_weight * deg[col]

# ###Rescale the Laplacian eigenvalues in [-1, 1]
# ##rescale: 2L/lmax-I; lmax=1.0
# fill_value = 1  ##-0.5
# edge_index, lap = add_self_loops(edge_index, -lap, fill_value, num_nodes)

# laplacian_matrix = sparse.coo_matrix((lap.numpy(),edge_index),shape=(num_nodes,num_nodes))


if __name__ == "__main__":
    # Parameters definition
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    conn_dir = os.path.join(data_dir, "processed",
                            "cobre_difumo512", "connectomes")
    ts_dir = os.path.join(data_dir, "processed",
                          "cobre_difumo512", "timeseries")
    raw_dir = os.path.join(data_dir, "raw")
    pheno_path = os.path.join(raw_dir, "cobre", "phenotypic_data.tsv")
    num_parcels = 512
    patient_no = 0

    # Data loading
    atlas = nil.datasets.fetch_atlas_difumo(
        data_dir=raw_dir, dimension=num_parcels)
    data = nil.datasets.fetch_cobre(data_dir=raw_dir, n_subjects=None)
    RawDataLoad = simexp_gcn.data.raw_data_loader.RawDataLoader(
        num_nodes=num_parcels, ts_dir=ts_dir, conn_dir=conn_dir, pheno_path=pheno_path)
    connectomes = RawDataLoad.get_valid_connectomes()
    graph = simexp_gcn.features.graph_construction.make_group_graph(
        connectomes, k=8, self_loops=False, symmetric=True)
    # Embedding error
    embedding_error(data.func[patient_no], graph,
                    atlas['maps'], confounds=data.confounds[patient_no])


# #input  should be a pytorch model
# model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "gcn_test.pt")

# model = torch.load(model_path)
# model.eval().state_dict()
# print(torch.max(gcn.forward(dat[5]), 1))
