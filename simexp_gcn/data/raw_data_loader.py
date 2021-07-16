import os
import warnings
import pandas as pd
import numpy as np
import simexp_gcn.data as data
import simexp_gcn.data.utils

class RawDataLoader():
  def __init__(self, num_nodes, ts_dir=None, conn_dir=None, pheno_path=None):
    """ Initializer for DataLoader class.

    Parameters
    ----------
      num_nodes: `int`
        Number of nodes of the graph (or number of parcelations).
      ts_dir: str
        path to directory w/ timeseries
      conn_dir: str
        path to directory w/ connectomes
      pheno_path: str
        path to phenotype file. corresponding file must be in .tsv format & columns ID and 'Subject Type'
    """
    self.num_nodes = num_nodes
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
    self.ts_shapes = self._get_ts_shapes()
    self.node_axis = (self.ts_shapes == self.num_nodes).all(axis=0)
    self.valid_ids = self._check_valid_ids()
    # filter phenotype and file lists
    mask_valid = np.in1d(self.ids, self.valid_ids)
    self.pheno = self.pheno[mask_valid]
    self.pheno = self.pheno.reset_index(drop=True)
    self.valid_ts_filepaths = np.array(self.ts_filepaths)[mask_valid]
    self.valid_conn_filepaths = np.array(self.conn_filepaths)[mask_valid]

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

  def _get_ts_shapes(self):
    """ Get the shapes of timeseries.
    """
    shapes = []
    # check all shapes from *.npy header
    for ts_filepath in self.ts_filepaths:
      shape = data.utils.read_npy_array_header(ts_filepath)[0]
      shapes += [shape]

    return np.array(shapes)

  def _check_valid_ids(self):
    """ Check whether participant IDs are valid regarding the timeserie data.
    """
    duration_axis = np.invert(self.node_axis)
    durations = self.ts_shapes[:, duration_axis].flatten()
    # check durations
    self.most_common_dura = np.bincount(durations).argmax()
    mask_valid_ts = (durations == self.most_common_dura)
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
      idx: `int`, `list` of `int` or `slice`
        list of indices to load.
    """
    
    if not (isinstance(idx, list) | isinstance(idx, slice)):
      raise ValueError("Input idx must be a `list` of int, but is {}!".format(type(idx)))
    timeseries = []
    # Get valid timeseries (with correct shapes)
    for ts_filepath in self.valid_ts_filepaths[idx]:
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
    for conn_filepath in self.valid_conn_filepaths[idx]:
      connectomes += [np.load(conn_filepath)]

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
    labels = np.array(self.pheno['Subject Type'].map({'Patient':1,'Control':0}).tolist())
    labels = labels[idx]

    return labels

  def split_timeseries_and_save(self, window_length, zero_padding=True, output_dir=None):
    """ Split the timeseries into time windows of specified length, and save them with the corresponding label file.

    Parameters
    ----------
    window_length: `int`
      time window length for each split, if -1 then do not split but still save.
    zero_padding: `bool`
      pad with zeros if timeserie cannot be evenly splitted, if `False` then remove last split instead.
    output_dir: `string`
      temporary directory where to save splitted timeseries and label file.
    """
    #TODO: split from task event file
    #TODO: use joblib.Memory
    #TODO: Define behaviour for incomplete data
      #1. Extrapolate (current, pad with zeros)
      #2. drop incomplete
    if output_dir is None:
      output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "interim")
    else:
      if not os.path.exists(output_dir):
        raise ValueError("output_dir does not exists: {}".format(output_dir))

    label_df = pd.DataFrame(columns=['label', 'filename'])
    out_file = os.path.join(output_dir, "{}_{:03d}.npy")
    out_csv = os.path.join(output_dir, "labels.csv")
    ts_duration = self.most_common_dura

    for ii in range(len(self.valid_ts_filepaths)):
      ts_filename = os.path.basename(self.valid_ts_filepaths[ii])
      ts_filename = "".join(ts_filename.split(".")[:-1])
      ts_label = self.get_valid_labels([ii])[0]
      ts_data = self.get_valid_timeseries([ii])[0]
      # Split the timeseries
      rem = ts_duration % window_length
      if rem == 0:
        n_splits = int(ts_duration / window_length)
      else:
        if zero_padding:
          n_splits = np.ceil(ts_duration / window_length)
          pad_size = int(n_splits*window_length - ts_duration)
          pad_widths = [(0, pad_size), (0, 0)]
          ts_data = np.pad(ts_data, pad_width=pad_widths)
        else:
          ts_data = ts_data[:(ts_duration-rem), :]
          n_splits = np.floor(ts_duration / window_length)
      # save splitted timeserie and label
      for jj, split_ts in enumerate(np.split(ts_data, n_splits)):
        ts_output_file = out_file.format(ts_filename, jj)
        # swap axis for pytorch compatibility if node is not the first axis
        if not self.node_axis[0]:
          split_ts = np.swapaxes(split_ts, 0, 1)
        np.save(ts_output_file, split_ts)
        curr_label = {'label': ts_label, 'filename': os.path.basename(ts_output_file)}
        label_df = label_df.append(curr_label, ignore_index=True)
    label_df.to_csv(out_csv, index=False)

if __name__ == "__main__":
  data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "cobre_difumo512")
  pheno_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "cobre", "phenotypic_data.tsv")
  
  RawDataLoad = RawDataLoader(
      num_nodes=512
      , ts_dir=os.path.join(data_dir, "timeseries")
      , conn_dir=os.path.join(data_dir, "connectomes")
      , pheno_path=pheno_path)
  timeseries = RawDataLoad.get_valid_timeseries()
  connectomes = RawDataLoad.get_valid_connectomes()
  labels = RawDataLoad.get_valid_labels()
  phenotype = RawDataLoad.get_valid_pheno()
  RawDataLoad.split_timeseries_and_save(window_length=50, zero_padding=True, output_dir=None)