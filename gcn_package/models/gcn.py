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
<<<<<<< HEAD
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
=======
>>>>>>> 694f25d81c03c19b49c8aa51cc8d2a00752098ad

        self.features = tg.nn.Sequential('x, edge_index, edge_weight', [
            (tg.nn.ChebConv(in_channels=n_timepoints, out_channels=n_filters, K=2,bias=True),
             'x, edge_index, edge_weight -> x'),
            nn.ReLU(inplace=True),
            (tg.nn.ChebConv(in_channels=n_filters, out_channels=n_filters, K=2, bias=True),
             'x, edge_index, edge_weight -> x'),
            nn.ReLU(inplace=True),
            (tg.nn.ChebConv(in_channels=n_filters, out_channels=n_filters, K=2, bias=True),
             'x, edge_index, edge_weight -> x'),
            nn.ReLU(inplace=True),
            (tg.nn.ChebConv(in_channels=n_filters, out_channels=n_filters, K=2, bias=True),
             'x, edge_index, edge_weight -> x'),
            nn.ReLU(inplace=True),
            (tg.nn.ChebConv(in_channels=n_filters, out_channels=n_filters, K=2, bias=True),
             'x, edge_index, edge_weight -> x'),
            nn.ReLU(inplace=True),
            (tg.nn.ChebConv(in_channels=n_filters, out_channels=n_filters, K=2, bias=True),
             'x, edge_index, edge_weight -> x'),
        ])
        self.pool = tg.nn.global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(n_filters * n_roi, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

<<<<<<< HEAD
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
=======
    def forward(self, x):
        batch_vector = torch.from_numpy(np.array(range(x.size(0)), dtype=int))
>>>>>>> 694f25d81c03c19b49c8aa51cc8d2a00752098ad

        x = self.features(x, self.edge_index, self.edge_weight)
        x = self.pool(x, batch_vector)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ChebConvBlock(torch.nn.Module):
    """ Custom ChebConvBlock """

    def __init__(self, edge_index, edge_attribute, gcn_filter_size=2, in_filters=32, out_filters=32,
                 gcn_normalization=None, bias=True, activation="ReLU", dropout=0.2, use_edge_pooling=False):
        super(self.__class__, self).__init__()
        self.edge_index = edge_index
        self.edge_attribute = edge_attribute
        self.dropout_threshold = dropout
        self.use_edge_pooling = use_edge_pooling
        # edge pooling https://graphreason.github.io/papers/17.pdf
        self.edge_pool = tg.nn.EdgePooling(
            in_channels=in_filters, dropout=self.dropout_threshold)
        self.cheb_conv = tg.nn.ChebConv(in_channels=in_filters, out_channels=out_filters,
                                        K=gcn_filter_size, normalization=gcn_normalization, bias=bias)
        self.activation = eval(f"torch.nn.{activation}()")
        self.dropout = torch.nn.Dropout(p=self.dropout_threshold)

    def forward(self, x):
        x = self.cheb_conv(x, edge_index=self.edge_index,
                           edge_weight=self.edge_attribute)
        x = self.activation(x)
        if self.dropout_threshold > 0:
            x = self.dropout(x)
        # TODO: issues with edge pooling
        # if self.use_edge_pooling:
        #     batch_vector = torch.from_numpy(
        #         np.array(range(x.size(0)), dtype=int))
        #     x, edge_index, _, _ = self.edge_pool(
        #         x, edge_index=self.edge_index, batch=batch_vector)

        return x


class LinearBlock(torch.nn.Module):
    """ Custom LinearBlock """

    def __init__(self, in_filters, out_filters, activation="ReLU", batch_normalisation=False, bias=True, dropout=0.2):
        super(self.__class__, self).__init__()
        self.batch_normalisation = batch_normalisation
        self.bias = bias
        self.dropout_threshold = dropout
        self.fc = torch.nn.Linear(in_filters, out_filters, bias=self.bias)
        self.activation = eval(f"torch.nn.{activation}()")
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


# for dynamic models: https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/building.html
# another example with gcn: https://gitcode.net/mirrors/dmlc/dgl/-/blob/master/examples/pytorch/seal/model.py
class CustomGCN(torch.nn.Module):
    def __init__(self, edge_index, edge_weight, n_timepoints=50, n_roi=512, n_classes=2,
                 n_gcn_layers=6, gcn_filters=32, gcn_filter_size=2, gcn_filter_decreasing_rate=1, gcn_normalization=None,
                 n_linear_layers=3, linear_filters=256, linear_filter_decreasing_rate=2, batch_normalisation=False,
                 bias=True, activation="ReLU", dropout=0.2):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_timepoints = n_timepoints
        self.n_roi = n_roi
        self.n_gcn_layers = n_gcn_layers
        self.gcn_filters = gcn_filters
        self.gcn_filter_size = gcn_filter_size
        self.gcn_filter_decreasing_rate = gcn_filter_decreasing_rate
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv
        self.gcn_normalization = gcn_normalization
        self.n_linear_layers = n_linear_layers
        self.linear_filters = linear_filters
        self.linear_filter_decreasing_rate = linear_filter_decreasing_rate
        self.bias = bias
        # activations: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        self.activation = activation
        self.batch_normalisation = batch_normalisation
        self.dropout = dropout
        self.n_classes = n_classes
        # build encoder block (GCN)
        self.encoder = torch.nn.ModuleList()
        for ii in range(self.n_gcn_layers):
            in_encode_filters = int(self.gcn_filters // self.gcn_filter_decreasing_rate**(ii-1))
            out_encode_filters = int(self.gcn_filters // self.gcn_filter_decreasing_rate**ii)
            if ii == 0:
                in_encode_filters = self.n_timepoints
                out_encode_filters = self.gcn_filters
            self.encoder.append(ChebConvBlock(edge_index=self.edge_index, edge_attribute=self.edge_weight, gcn_filter_size=self.gcn_filter_size,
                                              gcn_normalization=self.gcn_normalization, bias=self.bias, activation=self.activation, dropout=self.dropout,
                                              in_filters=in_encode_filters, out_filters=out_encode_filters))
        # build decoder block (linear layers)
        self.decoder = torch.nn.ModuleList()
        for ii in range(self.n_linear_layers):
            in_decode_filters = int(
                self.linear_filters // self.linear_filter_decreasing_rate**(ii-1))
            out_decode_filters = int(
                self.linear_filters // self.linear_filter_decreasing_rate**ii)
            if ii == 0:
                in_decode_filters = out_encode_filters * self.n_roi
                out_decode_filters = self.linear_filters
            if ii == (self.n_linear_layers - 1):
                out_decode_filters = self.n_classes
            self.decoder.append(LinearBlock(activation=self.activation, batch_normalisation=self.batch_normalisation, bias=self.bias,
                                            dropout=self.dropout, in_filters=in_decode_filters, out_filters=out_decode_filters))

    def forward(self, x):
        batch_vector = torch.range(start=0, end=x.size(0), dtype=int)
        # gcn encoder
        for encode_layer in self.encoder:
            x = encode_layer(x)
        # global mean pool and flattening
        x = tg.nn.global_mean_pool(x, batch=batch_vector)
        x = torch.flatten(x, 1)
        # lineard decoder
        for decode_layer in self.decoder:
            x = decode_layer(x)
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
