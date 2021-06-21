import numpy.lib.format

# https://stackoverflow.com/questions/64226337/is-there-a-way-to-read-npy-header-without-loading-the-whole-file
def read_npy_array_header(filepath):
    with open(filepath, 'rb') as fobj:
      version = numpy.lib.format.read_magic(fobj)
      func_name = 'read_array_header_' + '_'.join(str(v) for v in version)
      func = getattr(numpy.lib.format, func_name)
      header = func(fobj)
    
    return header

######################
# GRAPH CONSTRUCTION #
######################

def make_undirected(mat):
    """Takes an input adjacency matrix and makes it undirected (symmetric)."""
    m = mat.copy()
    mask = mat != mat.transpose()
    vals = mat[mask] + mat.transpose()[mask]
    m[mask] = vals
    return m

def knn_graph(mat,k=8):
    """Takes an input matrix and returns a k-Nearest Neighbour weighted adjacency matrix."""
    is_undirected = (mat == mat.T).all()
    m = np.abs(mat.copy())
    np.fill_diagonal(m,0)
    slices = []
    for i in range(m.shape[0]):
        s = m[:,i]
        not_neighbours = s.argsort()[:-k]
        s[not_neighbours] = 0
        slices.append(s)
    if is_undirected:
        return np.array(slices)
    else:
        return make_undirected(np.array(slices))
    
def make_group_graph(conn_path,k=8):
  """conn_path: path to dir containing connectomes, must be in .npy format."""
    # Load connectomes
    connectomes = [np.load(os.path.join(conn_path,p)) for p in os.listdir(conn_path)]

    # Group average connectome
    avg_conn = np.array(connectomes).mean(axis=0)

    # Undirected 8 k-NN graph as matrix
    avg_conn8 = knn_graph(avg_conn,k=k)

    # Format matrix into graph for torch_geometric
    graph = nx.convert_matrix.from_numpy_array(avg_conn8)
    return tg.utils.from_networkx(graph)

################
# TIME WINDOWS #
################

def split_timeseries(ts,n_timepoints=50):
    """Takes an input timeseries and splits it into time windows of specified length. Need to choose a number that splits evenly."""
    if ts.shape[0] % n_timepoints != 0:
        raise ValueError('Yikes choose a divisor for now')
    else:
        n_splits = ts.shape[0] / n_timepoints
        return np.split(ts,n_splits)

def split_ts_labels(timeseries,labels,n_timepoints=50):
    """
    timeseries: list of timeseries
    labels: list of lists (of accompanying labels)
    n_timepoints: n_timepoints of split (must be an even split)
    """
    # Split the timeseries
    split_ts = []
    tmp = [split_timeseries(t,n_timepoints=n_timepoints) for t in timeseries]
    for ts in tmp:
        split_ts = split_ts + ts

    #keep track of the corresponding labels
    n = int(timeseries[0].shape[0]/n_timepoints)
    split_labels = []
    for l in labels:
        split_labels.append(np.repeat(l,n))

    #add a label for each split
    split_labels.append(list(range(n))*len(timeseries))
    return split_ts, split_labels

def train_test_val_splits(split_ids,test_size=0.20,val_size=0.10,random_state=111):
    """Train test val split the data (in splits) so splits from a subject are in the same group.
        returns INDEX for each split
    """
    # Train test validation split of ids, then used to split dataframe
    X = np.unique(split_ids)
    y = list(range(len(X)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size+val_size, random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size/(test_size+val_size), random_state=random_state)

    train_idx = []
    test_idx = []
    val_idx = []
    for i in range(len(split_ids)):
        if split_ids[i] in X_train:
            train_idx.append(i)
        elif split_ids[i] in X_test:
            test_idx.append(i)
        elif split_ids[i]in X_val:
            val_idx.append(i)

    return train_idx,test_idx,val_idx