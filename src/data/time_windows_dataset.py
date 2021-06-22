import torch
from torch.utils.data import Dataset
from src.data import utils
from src.features import graph_construction as graph

class TimeWindows(Dataset):
    def __init__(self,timeseries,connectomes,sub_ids,labels,test_size=0.20,val_size=0.10,random_state=111,n_timepoints=50,k=8):
        """
        timeseries: list of arrays
            List of timeseries, all timeseries must be of the same length.
        connectomes: list of arrays
            List of connectomes.
        sub_ids: list of int
            List of subject ids, assumed to be in the same order as connectomes, timeseries, and labels.
        labels: list of int
            List of labels.
        """
        # TODO: check if input is valid

        self.timeseries = timeseries
        self.connectomes = connectomes
        self.sub_ids = sub_ids
        self.labels = labels

        #make group connectome graph
        self.graph = graph.make_group_graph(self.connectomes,k=k)

        #split timeseries
        self.split_timeseries,split_labs = utils.split_ts_labels(self.timeseries,[self.sub_ids,self.labels],n_timepoints=n_timepoints)
        self.split_sub_ids = split_labs[0]
        self.split_labels = split_labs[1]
        self.split_ids = split_labs[-1]

        #train test val split the data (each sub's splits in one category only)
        self.train_idx,self.test_idx,self.val_idx = utils.train_test_val_splits(self.split_sub_ids,
                                                                            test_size=test_size,
                                                                            val_size=val_size,
                                                                            random_state=random_state)

    def __len__(self):
        return len(self.split_sub_ids)

    def __getitem__(self,idx):
        ts = torch.from_numpy(self.split_timeseries[idx]).transpose(0,1)
        sub_id = self.split_sub_ids[idx]
        label = self.split_labels[idx]
        split_id = self.split_ids[idx]
        return ts,label