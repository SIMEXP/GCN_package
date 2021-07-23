#TODO: estimate the connectome from a list of data
  # 1. from atlas, with difumo
  # 2. Check if user want to denoise confounds --confound
    # a. if --confound-path: check if exists and use it
    # b. else: use confound associated with current subject
  # 3. for each file, run masker.fit_transform
    # a. check confound associated with current file
  # 4. estimate the connectome

#TODO: estimate connectomes from http://proceedings.mlr.press/v51/kalofolias16.pdf instead of correlation