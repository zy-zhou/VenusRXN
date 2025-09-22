# VenusRXN
Source code and dataset of VenusRXN.

See env.txt to install the python packages. The reaction-enzyme dataset for model training and evaluation is under
data/ folder.

# Data preprocess
run `python preprocess.py` to preprocess reaction SMILES and enzyme sequences.

# Test enzyme retrieval performance on the test set
run `python enzyme_retrieval.py` for reaction-enzyme retrieval

run `python enzyme_retrieval.py -ref` for enzyme-enzyme retrieval

