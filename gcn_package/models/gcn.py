import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
# import torch_geometric_temporal as tgt
import numpy as np


class YuGCN(torch.nn.Module):
    def __init__(self, edge_index, edge_weight, n_filters=32, n_roi=512, n_timepoints=50, n_classes=2):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.conv1 = tg.nn.ChebConv(
            in_channels=n_timepoints, out_channels=n_filters, K=2, bias=True)
        self.conv2 = tg.nn.ChebConv(
            in_channels=n_filters, out_channels=n_filters, K=2, bias=True)
        self.conv3 = tg.nn.ChebConv(
            in_channels=n_filters, out_channels=n_filters, K=2, bias=True)
        self.conv4 = tg.nn.ChebConv(
            in_channels=n_filters, out_channels=n_filters, K=2, bias=True)
        self.conv5 = tg.nn.ChebConv(
            in_channels=n_filters, out_channels=n_filters, K=2, bias=True)
        self.conv6 = tg.nn.ChebConv(
            in_channels=n_filters, out_channels=n_filters, K=2, bias=True)
        self.fc1 = nn.Linear(n_filters * n_roi, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.conv1(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.conv3(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.conv4(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.conv5(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.conv6(x, self.edge_index, self.edge_weight)
        x = tg.nn.global_mean_pool(x, torch.from_numpy(
            np.array(range(x.size(0)), dtype=int)))

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ChebConvBlock(nn.Module):
    """ Custom ChebConvBlock """

    def __init__(self, edge_index, edge_weight, gcn_filter_size=2, in_filters=32, out_filters=32, normalization=None, bias=True, activation="ReLU", dropout=0.2):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.dropout_threshold = dropout
        self.conv = tg.nn.ChebConv(in_channels=in_filters, out_channels=out_filters,
                                   K=gcn_filter_size, normalization=normalization, bias=bias)
        # list of all activations https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        self.activation = eval(f"torch.nn.functionnal.{activation}()")
        self.dropout = torch.nn.Dropout(p=self.dropout_threshold)
        # https://graphreason.github.io/papers/17.pdf
        self.edge_pooling = tg.nn.EdgePooling(
            in_channels=in_filters, dropout=self.dropout_threshold)

    def forward(self, x):
        if self.dropout_threshold > 0:
            x = self.edge_pooling(x, self.edge_index)
        x = self.conv(x, self.edge_index, self.edge_weight)
        x = self.activation(x)
        if self.dropout_threshold > 0:
            x = self.dropout(x)

        return x


class LinearBlock(nn.Module):
    """ Custom LinearBlock """

    def __init__(self, in_filters, out_filters, activation="ReLU", batch_normalisation=False, dropout=0.2):
        self.dropout_threshold = dropout
        self.batch_normalisation = batch_normalisation
        self.fc = torch.nn.Linear(in_filters, out_filters)
        self.activation = eval(f"torch.nn.functionnal.{activation}")
        self.batch_norm = torch.nn.BatchNorm1d(num_features=out_filters)
        self.dropout = torch.nn.Dropout(p=self.dropout_threshold)

    def forward(self, x):

        x = self.fc(x)
        x = self.activation(x)
        if self.batch_normalisation:
            x = self.batch_norm(x)
        if self.dropout_threshold > 0:
            x = self.dropout(x)

        return x


class LoicGCN(torch.nn.Module):
    def __init__(self, edge_index, edge_weight, n_timepoints=50, n_classes=2):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.conv1 = tg.nn.ChebConv(
            in_channels=n_timepoints, out_channels=32, K=2, bias=True)
        self.conv2 = tg.nn.ChebConv(
            in_channels=32, out_channels=32, K=2, bias=True)
        self.conv3 = tg.nn.ChebConv(
            in_channels=32, out_channels=16, K=2, bias=True)
        self.fc1 = nn.Linear(512*16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.2)
        # adding persistent buffer for edges serialization
        # self.register_buffer('edge_index', edge_index)
        # self.register_buffer('edge_weight', edge_weight)

    def forward(self, x):
        x = self.conv1(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = tg.nn.global_mean_pool(x, torch.from_numpy(
            np.array(range(x.size(0)), dtype=int)))
        x = x.view(-1, 512*16)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class STGCN(torch.nn.Module):
    def __init__(self, edge_index, edge_weight, n_timepoints=50, n_classes=2):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.conv1 = tg.nn.ChebConv(
            in_channels=n_timepoints, out_channels=32, K=2, bias=True)
        self.conv2 = tg.nn.ChebConv(
            in_channels=32, out_channels=32, K=2, bias=True)
        self.conv3 = tg.nn.ChebConv(
            in_channels=32, out_channels=16, K=2, bias=True)
        # self.recurent = tgt.nn.recurrent.STConv(in_channels=16, out_channels=16, K=2, hidden_channels=4, kernel_size=10)
        self.fc1 = nn.Linear(512*16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.2)
        # adding persistent buffer for edges serialization
        # self.register_buffer('edge_index', edge_index)
        # self.register_buffer('edge_weight', edge_weight)

    def forward(self, x):
        x = self.conv1(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        # x = self.recurent(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        x = tg.nn.global_mean_pool(x, torch.from_numpy(
            np.array(range(x.size(0)), dtype=int)))
        x = x.view(-1, 512*16)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x
