import os
import torch
import copy
import sklearn
import matplotlib.pyplot as plt

#TODO provide visualization tools
  #1. Basic graph plotting using https://networkx.org/documentation/stable/tutorial.html
  #2. Eigen vector representation: https://persagen.com/files/misc/arxiv-1712.00468.png
  #3. Basic clustering: https://github.com/neurolibre/gcn_tutorial_test
  #4. GCN latent space representation (T-SNE or PCA)
  #5. Graph signal reconstruction (for graph-AE)

def PCA(data):
  """
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
  """
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
    axes.scatter(x=pca_results[mask, 0], y=pca_results[mask, 1], label=f"class {curr_label}")
  axes.legend()
  axes.set_title("PCA decomposition of {} activation".format(layer_name))
  plt.show()


def vizualize_weights(model, layer_name):
  """
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

# #input  should be a pytorch model
# model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "gcn_test.pt")

# model = torch.load(model_path)
# model.eval().state_dict()
# print(torch.max(gcn.forward(dat[5]), 1))