import os
import warnings
import re
import pandas as pd
import numpy as np
import utils

class DataLoader():
  def __init__(self, ts_dir=None, conn_dir=None, pheno_path=None):
    
    self.ts_dir = ts_dir
    self.conn_dir = conn_dir
    self.pheno_path = pheno_path
    # check directories existence
    if not os.path.exists(self.ts_dir):
      raise ValueError("ts_dir does not exists: {}".format(self.ts_dir))
    if not os.path.exists(self.conn_dir):
      raise ValueError("conn_dir does not exists: {}".format(self.conn_dir))
    if not os.path.exists(self.pheno_path):
      raise ValueError("pheno_path does not exists: {}".format(self.pheno_path))
    # get list of files
    self.list_ts_files = sorted(os.listdir(self.ts_dir))
    self.list_conn_files = sorted(os.listdir(self.conn_dir))
    self.non_valid_ids = self._check_non_valid_ids()

  def _check_non_valid_ids(self):

    durations = []
    ids = []
    pattern = ".*?_([0-9]+)_.*\\.npy"
    # check all shapes from *.npy header and read IDs
    for ts_file in self.list_ts_files:
      ts_filepath = os.path.join(self.ts_dir, ts_file)
      shape = utils.read_npy_array_header(ts_filepath)[0]
      durations += [shape[0]]
      ids += [re.match(pattern, ts_filepath)[1]]
    # check durations
    most_common_dura = np.bincount(durations).argmax()
    mask_valid_ts = (durations == most_common_dura)
    non_valid_ids = np.array(ids)[~mask_valid_ts]
    if not mask_valid_ts.all():
      warnings.warn("Different shapes for sub ID(s): {}".format(non_valid_ids))

    return non_valid_ids

  def get_timeseries(self):

    timeseries = []
    # Get valid timeseries (with correct shapes)
    for ts_file in self.list_ts_files:
      is_valid = (np.char.find(ts_file, self.non_valid_ids) <= 0)
      if is_valid:
        ts_filepath = os.path.join(self.ts_dir, ts_file)
        timeseries += [np.load(ts_filepath)]

    return timeseries

  def get_connectomes(self):

    connectomes = []
    # load connectomes
    for conn_file in self.list_conn_files:
      is_valid = (np.char.find(conn_file, self.non_valid_ids) <= 0)
      if is_valid:
        connectomes += [np.load(os.path.join(self.conn_dir, conn_file))]

    return connectomes

  def get_pheno_labels(self):
    
    pheno = pd.read_csv(self.pheno_path, delimiter='\t')
    ids_pheno = np.array(pheno['ID'], dtype=str)
    non_valid_ids = np.where(np.in1d(ids_pheno, self.non_valid_ids))[0]
    pheno = pheno.drop(labels=non_valid_ids, axis=0)
    pheno.sort_values('ID',inplace=True)
    pheno = pheno.reset_index(drop=True)
    labels = pheno['Subject Type'].map({'Patient':1,'Control':0}).tolist()

    return pheno, labels

if __name__ == "__main__":
  data_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cobre_difumo512", "difumo")
  
  Data = DataLoader(
    ts_dir = os.path.join(data_dir, "timeseries")
    , conn_dir = os.path.join(data_dir, "connectomes")
    , pheno_path = os.path.join(data_dir, "phenotypic_data.tsv"))
  timeseries = Data.get_timeseries()
  connectomes = Data.get_connectomes()
  phenotype, labels = Data.get_pheno_labels()
