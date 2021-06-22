import numpy.lib.format
import numpy as np
from sklearn.model_selection import train_test_split

# https://stackoverflow.com/questions/64226337/is-there-a-way-to-read-npy-header-without-loading-the-whole-file
def read_npy_array_header(filepath):
    with open(filepath, 'rb') as fobj:
      version = numpy.lib.format.read_magic(fobj)
      func_name = 'read_array_header_' + '_'.join(str(v) for v in version)
      func = getattr(numpy.lib.format, func_name)
      header = func(fobj)
    
    return header

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