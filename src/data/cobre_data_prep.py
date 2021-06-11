from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
import os
import numpy as np

print('Fetching atlas...')
atlas = datasets.fetch_atlas_difumo(dimension=512)
atlas_filename = atlas['maps']
atlas_labels = atlas['labels']
print('Done fetching atlas.')

print('Fetching data...')
data = datasets.fetch_cobre(n_subjects = None) #all subs
print('Done fetching data.')

masker = NiftiMapsMasker(maps_img = atlas['maps'],standardize=True,verbose=5)
corr_measure = ConnectivityMeasure(kind='correlation')

print('Starting extraction...')
ts_out = '/home/harveyaa/Documents/fMRI/data/cobre/difumo/timeseries'
conn_out = '/home/harveyaa/Documents/fMRI/data/cobre/difumo/connectomes'
for i in range(len(data.func)):
    ts = masker.fit_transform(data.func[i],confounds=data.confounds[i])
    conn = corr_measure.fit_transform([ts])[0]
    sub_num = os.path.basename(data.func[i]).split('.')[0].split('_')[1]
    np.save(os.path.join(ts_out,'timeseries_{}_difumo_512.npy'.format(sub_num)),ts)
    np.save(os.path.join(conn_out,'conn_{}_difumo_512.npy'.format(sub_num)),conn)
    print('Done {}/146'.format(i+1))
