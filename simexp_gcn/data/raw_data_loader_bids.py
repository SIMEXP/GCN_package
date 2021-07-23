#TODO: BIDS detection
  #1 first detect if input data is BIDS, and was preprocessed with fmriprep
    # a. if BIDS but not fmriprep: ask user to preprocess OR povide folder with timeseries + connectomes + events (optionnally)
    # b. else: ask user for BIDS OR povide folder with timeseries + connectomes + events (optionnally)
  #2 in case user provide timeseries + connectomes + events folder, skip this module
#TODO: detect BIDS file (with pybids query), depending on bids filter
  # 1. filter all files based on .bidsfilters (if user wants to use just one participant, or one condition for example)
  # 2. detect all subjects and associated events (if exists) with pybids query (to be agnostic to BIDS version)
#TODO: call features/connectome_generator
#TODO: prepare data and labels from --prediction or --autoencoding
  # 1. let user choose whether it will perfom a prediction or encoding
    # a. if --prediction: read participants.tsv to get prediction target (user choice for which column to use, for example "sex", "group" or	"age")
    # b. if --autoencoder: skip 
