import os
import numpy as np
import pandas as pd
import src.data.time_windows_dataset as tw
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from src.models.yu_gcn import YuGCN
from src.models import utils
from src.features import graph_construction as graph
import pytest

def cobre_test_data(n_subs=10):
    #data_path = os.path.join("..", "data", "cobre_difumo512")
    data_path = '/home/harveyaa/Documents/fMRI/data/cobre'
    ts_path = os.path.join(data_path, "difumo", "timeseries")
    conn_path = os.path.join(data_path, "difumo", "connectomes")
    pheno_path = os.path.join(data_path, "difumo", "phenotypic_data.tsv")

    pheno = pd.read_csv(pheno_path,delimiter='\t')
    pheno = pheno[pheno['ID']!=40075]
    pheno.sort_values('ID',inplace=True)
    labels = pheno['Subject Type'].map({'Patient':1,'Control':0}).tolist()

    timeseries = [np.load(os.path.join(ts_path,p)) for p in sorted(os.listdir(ts_path))[:n_subs]]
    sub_ids = [int(p.split('_')[1]) for p in sorted(os.listdir(ts_path))[:n_subs]]
    connectomes = [np.load(os.path.join(conn_path,p)) for p in sorted(os.listdir(conn_path))[:n_subs]]

    if 40075 in sub_ids:
        idx = sub_ids.index(40075)
        del sub_ids[idx]
        del timeseries[idx]
        del connectomes[idx]
    return timeseries,connectomes,sub_ids,labels[:n_subs]

def fake_data(n_roi=15,n_timepoints=150,n_subs=10,seed=111):
    np.random.seed(seed)
    timeseries = [np.random.randn(n_timepoints,n_roi) for i in range(n_subs)]
    connectomes = []
    for i in range(n_subs):
        A = np.tril(np.random.randn(n_roi,n_roi))
        connectomes.append(A + np.transpose(A))
    sub_ids = np.random.choice(list(range(n_subs*2)),size=n_subs,replace=False).tolist()
    labels = np.random.binomial(1,0.5,size=n_subs).tolist()
    return timeseries,connectomes,sub_ids,labels

class TestFeatures:
    def test_make_undirected_valid_unweighted(self):
        A = np.array([[1,0,0],[1,0,0],[1,0,0]])
        B = np.array([[1,1,1],[1,0,0],[1,0,0]])
        assert (graph.make_undirected(A) == B).all()
    
    def test_make_undirected_valid_weighted(self):
        A = np.array([[0.75,0,0],[0.3,0,0],[0.6,0,0]])
        B = np.array([[0.75,0.15,0.3],[0.15,0,0],[0.3,0,0]])
        assert (graph.make_undirected(A) == B).all()
    
    def test_make_undirected_invalid(self):
        A = np.random.randn(2,3)
        with pytest.raises(ValueError):
            graph.make_undirected(A)
    
    def test_knn_graph_valid(self):
        A = np.array([[1.6,0.15,-0.1,-0.7],[0.15,-2.3,0.75,-0.6],[-0.1,0.75,1.5,-0.5],[-0.7,-0.6,-0.5,-1.1]])
        B = np.array([[0.,0.075,0.,-0.7],[0.075,0.,0.75,-0.6],[0,0.75,0.,-0.25],[-0.7,-0.6,-0.25,0.]])
        sol = graph.knn_graph(A,k=2)
        assert (sol == B).all()

class TestData:
    def test_timewindows_init_cobre(self):
        ts, conn, ids, labs = cobre_test_data()
        data = tw.TimeWindows(ts,conn,ids,labs)
        assert len(data.timeseries) == 10
    
    def test_timewindows_init_fake(self):
        ts, conn, ids, labs = fake_data()
        data = tw.TimeWindows(ts,conn,ids,labs)
        assert len(data.timeseries) == 10
    
    def test_timewindows_len_fake(self):
        ts, conn, ids, labs = fake_data()
        data = tw.TimeWindows(ts,conn,ids,labs)
        assert len(data) == 30

class TestModels:
    def test_yu_gcn(self):
        n_timepoints = 50
        batch_size = 128

        ts, conn, ids, labs = cobre_test_data()
        data = tw.TimeWindows(ts,conn,ids,labs,n_timepoints=n_timepoints)

        # Create PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(data.train_idx)
        test_sampler = SubsetRandomSampler(data.test_idx)
        val_sampler = SubsetRandomSampler(data.val_idx)

        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler)
        val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sampler)

        # Create model
        gcn = YuGCN(data.graph.edge_index,data.graph.weight,n_timepoints=n_timepoints)

        # Train and evaluate model
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001, weight_decay= 0.0005)

        epochs = 2
        for t in range(epochs):
            #print(f"Epoch {t+1}\n-------------------------------")
            utils.train_loop(train_loader, gcn, loss_fn, optimizer)
            utils.test_loop(test_loader, gcn, loss_fn)
        #print("Done!")