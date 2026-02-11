# VenusRXN
Source code and dataset of VenusRXN.

First install the dependencies according to environment.yml. The reaction-enzyme dataset for model training and evaluation is under data/ folder.

# Data preprocess
run `python preprocess_rxns.py` to preprocess reaction SMILES and cache the reaction graphs.

# Test enzyme retrieval performance
run `python enzyme_retrieval.py` for reaction-enzyme retrieval

run `python enzyme_retrieval.py -ref` for enzyme-enzyme retrieval
