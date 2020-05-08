# Learning Fair Representations via an Adversarial Framework

See https://github.com/propublica/compas-analysis for more information about the COMPAS dataset used.

This Notebook contains more contextual information about the variables: https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb

See https://archive.ics.uci.edu/ml/datasets/Bank+Marketing for more information about the bank marketing dataset used.


This is the code for implementation/reproduction of experiments described in paper https://arxiv.org/abs/1904.13341. 

Please find in model.py the implementation of models and in train.py the training inferface. 

## Running on New Datasets

To run an experiment on a different dataset, make sure that the dataset is stored in a .csv file, with only numeric inputs, and add the file to the data/ folder. Then, adjust the following command line arguments used with run_dataset.py to fit the context:
* --data: name of .csv file in the dataset to run the experiment on
* --alpha: tradeoff hyperparameter between MSE loss and EMD approximation loss
* --ndim: number of dimensions desired for the generated representations
* --protected: name of the binary column used to indicate membership in a protected class in the .csv file
* --y: name of the outcome column in the .csv file
* --bins: number of bins to use to calculate EMD
* --n_iter: number of iterations to use while training

For example, if 'dataset.csv' is the name of your file, you can run:

python3 run_dataset.py --data dataset.csv --alpha 100 --protected gender --y outcome

As seen in the example above, all of the command line arguments are optional, and so if any of them are not specifed, then the default option, which are the values specific to the COMPAS dataset, will be used instead. These default settings are:
* --data:'compas_new.csv'
* --ndim: 30
* --alpha: 1000
* --protected: 'P'
* --y: 'Y'
* --bins: 2
* --n_iter: 20

