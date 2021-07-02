gcn_package
==============================

The lab package for GCN

### TODO:
 - [ ] Tests for graph construction functions
 - [ ] Tests for data util functions
 - [ ] Tests for TimeWindows dataset
 - [ ] Update notebook implementation to use src - (tutorial)
 - [ ] Document functions
 - [ ] Implement val loop
 - [ ] Implement error throwing/valid input checks
 - [ ] Implement padding in split timeseries
     - timeseries length can be flexible
     - window size won't need to be a divisor
 - [ ] Command line tool for end-to-end training (optionnal)
     - Auto-encoder (target is the input) or classification (label)
     - agnostic timeseries-splitting and labelling:
         - Read participant condition (label) from [paticipant file](https://bids-specification.readthedocs.io/en/stable/03-modality-agnostic-files.html#participants-file) or phenotype.
         - Split and task-label based from [task event file](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html#task-events)
         - file should contain at least id
 
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Scripts to fetch and process data
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Modules for custom pytorch datasets & utils.
    │   │   └── time_windows_dataset.py
    │   │   └── data_loader.py
    │   │   └── utils.py
    │   │
    │   ├── features       <- Modules for building features from data.
    │   │   └── graph_construction.py
    │   │
    │   ├── models         <- Modules for different model architectures & utils to run them.
    │   │   ├── yu_gcn.py
    │   │   └── utils.py
    │   │
    │   └── visualization  <- Modules to create exploratory and results oriented visualizations
    │       └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
