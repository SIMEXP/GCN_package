import os
import torch

#TODO provide visualization tools
  #1. Basic graph plotting using https://networkx.org/documentation/stable/tutorial.html
  #2. Eigen vector representation: https://persagen.com/files/misc/arxiv-1712.00468.png
  #3. Basic clustering: https://github.com/neurolibre/gcn_tutorial_test
  #4. GCN latent space representation (T-SNE or PCA)
  #5. Graph signal reconstruction (for graph-AE)

#input 
model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "gcn_test.pt")

model = torch.load(model_path)
model.eval()
# print(torch.max(gcn.forward(dat[5]), 1))