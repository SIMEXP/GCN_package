import os
import numpy as np
import pandas as pd
import torch.utils.data
import raw_data

class cobreTimeWindows(torch.utils.data.Dataset):
    def __init__(self, ts_dir, pheno_path, test_size=0.20, val_size=0.10, random_seed=0, n_timepoints=50):
        self.pheno_path = pheno_path
        pheno = pd.read_csv(pheno_path,delimiter='\t')
        pheno = pheno[pheno['ID']!=40075]
        pheno.sort_values('ID',inplace=True)
        self.labels = pheno['Subject Type'].map({'Patient':1,'Control':0}).tolist()

        self.ts_path = ts_path
        self.timeseries = [np.load(os.path.join(ts_path,p)) for p in sorted(os.listdir(ts_path))]
        self.sub_ids = [int(p.split('_')[1]) for p in sorted(os.listdir(ts_path))]

        #filter out bad sub
        idx = self.sub_ids.index(40075)
        del self.sub_ids[idx]
        del self.timeseries[idx]

        #split timeseries
        self.split_timeseries,split_labs = split_ts_labels(self.timeseries,[self.sub_ids,self.labels],n_timepoints=n_timepoints)
        self.split_sub_ids = split_labs[0]
        self.split_labels = split_labs[1]
        self.split_ids = split_labs[-1]

        #train test val split the data (each sub's splits in one category only)
        self.train_idx,self.test_idx,self.val_idx = train_test_val_splits(self.split_sub_ids,