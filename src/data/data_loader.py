import os
import warnings
import pandas as pd
import numpy as np
import src.data as data
import src.data.utils

class DataLoader():
  def __init__(self, ts_dir=None, conn_dir=None, pheno_path=None):
    """ Initializer for DataLoader class.

    Parameters
    ----------
      ts_dir: str
        path to directory w/ timeseries
      conn_dir: str
        path to directory w/ connectomes
      pheno_path: str
        path to phenotype file. corresponding file must be in .tsv format & columns ID and 'Subject Type'
    """
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

    # get list of participant IDs from phenotype file
    self.pheno = self._get_pheno()
    self.ids = np.array(self.pheno['ID'], dtype=str)
    # get list of files based on phenotype file
    self.ts_filepaths, self.conn_filepaths = self._get_files_list()
    # infer non valid participant IDs based from timeserie files content
    self.valid_ids = self._check_valid_ids()
    # filter phenotype and file lists
    mask_valid = np.in1d(self.ids, self.valid_ids)
    self.pheno = self.pheno[mask_valid]
    self.pheno = self.pheno.reset_index(drop=True)
    self.ts_filepaths = np.array(self.ts_filepaths)[mask_valid]
    self.conn_filepaths = np.array(self.conn_filepaths)[mask_valid]

  def _get_pheno(self):
    """ Load phenotype file.
    """
    pheno = pd.read_csv(self.pheno_path, delimiter='\t')
    pheno.sort_values('ID',inplace=True)
    pheno = pheno.reset_index(drop=True)

    return pheno

  def _get_files_list(self):
    """ Get list of timeseries and connectome files, ordered by ID from phenotype.
    """
    ts_filepaths = []
    conn_filepaths = []
    ts_filepaths_from_dir = sorted(os.listdir(self.ts_dir))
    conn_filepaths_from_dir = sorted(os.listdir(self.conn_dir))
    for sub_id in self.ids:
      for ts_file in ts_filepaths_from_dir:
        if sub_id in ts_file:
          ts_filepaths += [os.path.join(self.ts_dir, ts_file)]
          ts_filepaths_from_dir.remove(ts_file)
          break
      for conn_file in conn_filepaths_from_dir:
        if sub_id in conn_file:
          conn_filepaths += [os.path.join(self.conn_dir, conn_file)]
          conn_filepaths_from_dir.remove(conn_file)
          break

    return ts_filepaths, conn_filepaths

  def _check_valid_ids(self):
    """ Check whether participant IDs are valid regarding the timeserie data.
    """
    durations = []
    # check all shapes from *.npy header
    for ts_filepath in self.ts_filepaths:
      shape = data.utils.read_npy_array_header(ts_filepath)[0]
      durations += [shape[0]]
    # check durations
    most_common_dura = np.bincount(durations).argmax()
    mask_valid_ts = (durations == most_common_dura)
    non_valid_ids = np.array(self.ids)[~mask_valid_ts]
    if not mask_valid_ts.all():
      warnings.warn("Different shapes for sub ID(s): {}".format(non_valid_ids))

    return np.array(self.ids)[mask_valid_ts]
  
  def get_valid_pheno(self):
    """ Load valid and filtered phenotype file.
    """
    return self.pheno

  def get_valid_timeseries(self, idx=slice(None)):
    """ Load valid timeserie data.

    Parameters
    ----------
      idx: `list` of `int` or `slice`
        list of indices to load.
    """
    
    if not (isinstance(idx, list) | isinstance(idx, slice)):
      raise ValueError("Input idx must be a `list` of int, but is {}!".format(type(idx)))
    timeseries = []
    # Get valid timeseries (with correct shapes)
    for ts_file in self.ts_filepaths[idx]:
      is_valid = (np.char.find(ts_file, self.valid_ids) > 0).any()
      if is_valid:
        ts_filepath = os.path.join(self.ts_dir, ts_file)
        timeseries += [np.load(ts_filepath)]

    return timeseries

  def get_valid_connectomes(self, idx=slice(None)):
    """ Load valid connectomes.

    Parameters
    ----------
      idx: `list` of `int` or `slice`
        list of indices to load.
    """
    
    if not (isinstance(idx, list) | isinstance(idx, slice)):
      raise ValueError("Input idx must be a `list` of int, but is {}!".format(type(idx)))
    connectomes = []
    # load connectomes
    for conn_file in self.conn_filepaths[idx]:
      is_valid = (np.char.find(conn_file, self.valid_ids) > 0).any()
      if is_valid:
        connectomes += [np.load(os.path.join(self.conn_dir, conn_file))]

    return connectomes

  def get_valid_labels(self, idx=slice(None)):
    """ Load labels from phenotype file.
    
    Parameters
    ----------
      idx: `list` of `int` or `slice`
        list of indices to load.
    """
    
    if not (isinstance(idx, list) | isinstance(idx, slice)):
      raise ValueError("Input idx must be a `list` of int, but is {}!".format(type(idx)))
    labels = self.pheno['Subject Type'].map({'Patient':1,'Control':0}).tolist()
    labels = labels[idx]

    return labels

if __name__ == "__main__":
  data_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cobre_difumo512", "difumo")
  
  DataLoad = DataLoader(
    ts_dir = os.path.join(data_dir, "timeseries")
    , conn_dir = os.path.join(data_dir, "connectomes")
    , pheno_path = os.path.join(data_dir, "phenotypic_data.tsv"))
  timeseries = DataLoad.get_valid_timeseries()
  connectomes = DataLoad.get_valid_connectomes()
  labels = DataLoad.get_valid_labels()
  phenotype = DataLoad.get_valid_pheno()
