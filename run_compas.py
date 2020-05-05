import torch
import torch.nn as nzn
import torch.optim as optim
import torch.distributions as D
import os

import numpy as np
from pyemd import emd_samples

from model import FairRep
from helpers import update_progress, normalize, total_correlation, cal_emd_resamp
import time
import sys
from train import train_rep
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from dumb_containers import split_data, evaluate_performance_sim

import pandas as pd

np.random.seed(1)



# In[2]:


def split_data_np(data, ratio):
    data_train = []
    data_test = []
    split = int(len(list(data)[0]) * ratio)
    #print(list(data))
    for d in data:
        #print(d)
        data_train.append(d[:split])
        data_test.append(d[split+1:])
    return data_train, data_test

def sigmoid(X):
    return 1 / (1+np.exp(-X))

def get_consistency(X, classifier, n_neighbors, based_on=None):
    nbr_model = NearestNeighbors(n_neighbors=n_neighbors+1, n_jobs=-1)
    if based_on is None:
        based_on = X
    nbr_model.fit(based_on)
    _, indices = nbr_model.kneighbors(based_on)
    X_nbrs = X[indices[:, 1:]]
    knn_mean_scores = np.mean(sigmoid(X_nbrs.dot(classifier.coef_.T) + classifier.intercept_), axis=1)
    scores = sigmoid(X.dot(classifier.coef_.T) + classifier.intercept_)
    mean_diff = np.mean(np.abs(scores - knn_mean_scores))
    consistency = 1-mean_diff
    return consistency

def stat_diff(X, P, model):
    scores = sigmoid(X.dot(model.coef_.T) + model.intercept_)
    return np.abs(np.mean(scores[P==0]) - np.mean(scores[P==1]))

def shuffled_np(df):
    return np.random.shuffle(df.values)

def get_model_preds(X_train, y_train, P_train, X_test, y_test, P_test, model_name):
    lin_model = LogisticRegression(C=C, solver='sag', max_iter=2000)
    lin_model.fit(X_train, y_train)

    y_test_scores = sigmoid((X_test.dot(lin_model.coef_.T) + lin_model.intercept_).flatten())
    y_hats[model_name] = y_test_scores

    print('logistic regression evaluation...')
    performance = list(evaluate_performance_sim(y_test, y_test_scores, P_test))
    return lin_model, y_test_scores, performance

def get_preds_on_full_dataset(x_context, lin_model):
    return sigmoid(((x_context.numpy()).dot(lin_model.coef_.T) + lin_model.intercept_).flatten())

def test_in_one(n_dim, batch_size, n_iter, C, alpha,compute_emd=True, k_nbrs = 3, emd_method=emd_samples):
    global X, P, y, df, X_test

    reps = {}

    X_no_p = df.drop(['Y', 'P'], axis=1).values

    # declare variables
    X = torch.tensor(X).float()
    P = torch.tensor(P).long()
    # train-test split
    data_train, data_test = split_data_np((X.data.cpu().numpy(),P.data.cpu().numpy(),y), 0.7)
    X_train, P_train, y_train = data_train
    X_test, P_test, y_test = data_test
    X_train_no_p = X_train[:, :-1]
    X_test_no_p = X_test[:, :-1]
    X_u = X[P==1]
    X_n = X[P==0]

    # AE.
    model_ae = FairRep(len(X[0]), n_dim)
    train_rep(model_ae, 0.01, X, P, n_iter, 10, batch_size, alpha = 0, C_reg=0, compute_emd=compute_emd, adv=False, verbose=True)

    # AE_P.
    model_ae_P = FairRep(len(X[0])-1, n_dim-1)
    train_rep(model_ae_P, 0.01,X_no_p , P, n_iter, 10, batch_size, alpha = 0, C_reg=0, compute_emd=compute_emd, adv=False, verbose=True)

    # NFR.
    model_name = 'Original'
    model_nfr = FairRep(len(X[0]), n_dim)
    X = torch.tensor(X).float()
    P = torch.tensor(P).long()
    train_rep(model_nfr, 0.01, X, P, n_iter, 10, batch_size, alpha = alpha, C_reg=0, compute_emd=compute_emd, adv=True, verbose=True)
    results={}

    print('begin testing.')
    X_ori_np = X.data.cpu().numpy()
    # Original.
    print('logistic regression on the original...')
    lin_model, y_test_scores, performance = get_model_preds(X_train, y_train, P_train, X_test, y_test, P_test, model_name)
    y_hats[model_name] = get_preds_on_full_dataset(X, lin_model)
    reps[model_name] = None

    performance.append(emd_method(X_n, X_u))
    performance.append(get_consistency(X.data.cpu().numpy(), lin_model, n_neighbors=k_nbrs))
    performance.append(stat_diff(X.data.cpu().numpy(), P, lin_model))
    results[model_name] = performance

    # Original-P.
    model_name = 'Original-P'
    print('logistic regression on the original-P')
    lin_model, y_test_scores, performance = get_model_preds(X_train_no_p, y_train, P_train, X_test_no_p, y_test, P_test, model_name)
    y_hats[model_name] = get_preds_on_full_dataset(X[:, :-1], lin_model)
    reps[model_name] = None

    performance.append(emd_method(X_n[:,:-1], X_u[:,:-1]))
    print('calculating consistency...')
    performance.append(get_consistency(X[:,:-1].data.cpu().numpy(), lin_model,  n_neighbors=k_nbrs))
    print('calculating stat diff...')
    performance.append(stat_diff(X[:,:-1].data.cpu().numpy(), P, lin_model))
    results[model_name] = performance




    # use encoder
    model_name = 'AE'

    U_0 = model_ae.encoder(X[P==0]).data
    U_1 = model_ae.encoder(X[P==1]).data
    U = model_ae.encoder(X).data

    U_np = U.cpu().numpy()
    data_train, data_test = split_data_np((U_np,P.data.cpu().numpy(),y), 0.7)
    X_train, P_train, y_train = data_train
    X_test, P_test, y_test = data_test

    print('logistic regression on AE...')
    lin_model = LogisticRegression(C=C, solver='sag', max_iter=2000)
    lin_model.fit(X_train, y_train)

    y_test_scores = sigmoid((X_test.dot(lin_model.coef_.T) + lin_model.intercept_).flatten())
    y_hats[model_name] = get_preds_on_full_dataset(U, lin_model)
    reps[model_name] = U

    def calc_perf(y_test, y_test_scores, P_test, U_0, U_1, U_np, lin_model, X_test):
        print('logistic regression evaluation...')
        performance = list(evaluate_performance_sim(y_test, y_test_scores, P_test))
        print('calculating emd...')
        performance.append(emd_method(U_0, U_1))
        print('calculating consistency...')
        performance.append(get_consistency(U_np, lin_model, n_neighbors=k_nbrs, based_on=X_ori_np))
        print('calculating stat diff...')
        performance.append(stat_diff(X_test, P_test, lin_model))
        return performance

    performance = calc_perf(y_test, y_test_scores, P_test, U_0, U_1, U_np, lin_model, X_test)
    results[model_name] = (performance)



    # AE minus P
    model_name = 'AE_P'
    U_0 = model_ae_P.encoder(X[:,:-1][P==0]).data
    U_1 = model_ae_P.encoder(X[:,:-1][P==1]).data
    U = model_ae_P.encoder(X[:,:-1]).data
    print('ae-p emd afterwards: ' + str(emd_method(U_0, U_1)))
    U_np = U.cpu().numpy()
    data_train, data_test = split_data_np((U_np,P.data.cpu().numpy(),y), 0.7)
    X_train, P_train, y_train = data_train
    X_test, P_test, y_test = data_test

    print('logistic regression on AE-P...')
    lin_model = LogisticRegression(C=C, solver='sag', max_iter=2000)
    lin_model.fit(X_train, y_train)

    y_test_scores = sigmoid((X_test.dot(lin_model.coef_.T) + lin_model.intercept_).flatten())
    y_hats[model_name] = get_preds_on_full_dataset(U, lin_model)
    reps[model_name] = U


    performance = calc_perf(y_test, y_test_scores, P_test, U_0, U_1, U_np, lin_model, X_test)
    results[model_name] = (performance)

    model_name = 'NFR'
    U_0 = model_nfr.encoder(X[P==0]).data
    U_1 = model_nfr.encoder(X[P==1]).data
    U = model_nfr.encoder(X).data
    print('nfr emd afterwards: ' + str(emd_method(U_0, U_1)))

    U_np = U.cpu().numpy()
    data_train, data_test = split_data_np((U_np,P.data.cpu().numpy(),y), 0.7)
    X_train, P_train, y_train = data_train
    X_test, P_test, y_test = data_test
    print('logistic regression on NFR...')
    lin_model = LogisticRegression(C=C, solver='sag', max_iter=2000)
    lin_model.fit(X_train, y_train)

    y_test_scores = sigmoid((X_test.dot(lin_model.coef_.T) + lin_model.intercept_).flatten())
    y_hats[model_name] = get_preds_on_full_dataset(U, lin_model)
    reps[model_name] = U

    performance = calc_perf(y_test, y_test_scores, P_test, U_0, U_1, U_np, lin_model, X_test)
    results[model_name] = (performance)

    return results, y_hats, reps

def save_predictions(y, y_hat, reps, model_name):
    # make CSV dataframe to store predicted scores
    y_hat = y_hat.reshape(len(X), 1)
    y = y.reshape(len(X), 1)

    data_yhat = np.concatenate((X, y, y_hat), axis=1)
    cols = list(df.columns)
    cols.append('y_hat')
    pred_df = pd.DataFrame(data = data_yhat, columns = cols)
    pred_df.to_csv('results/compas_preds_' + model_name + '.csv')

    if torch.is_tensor(reps):
        reps_np = reps.numpy()
        data_reps = np.concatenate((reps_np, y, y_hat), axis=1)
        num_cols = reps_np.shape[1]
        cols = []
        for i in range(num_cols):
            cols.append('repr_' + str(i))
        cols += ['y', 'y_hat']
        repr_df = pd.DataFrame(data = data_reps, columns = cols)
        repr_df.to_csv('results/compas_representation_' + model_name + '.csv')



# two batch of samples: one normal(0,1), and one uniform(0,1).
# with open('data/german.numeric.processed') as f:
    # data_raw = np.array([list(map(float, x)) for x in map(lambda x: x.split(), f)])
    # print('raw data')
    # print(data_raw)
    # data_raw = np.array(data_raw)
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'compas_clean.csv')
#filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'german_clean.csv')
try:
    df = pd.read_csv(filepath)
except IOError as err:
    print('IO error')

print('num COLs:')
print(len(list(df.columns)))

P = df['P'].values
y = df['Y'].values

# X contains protected class P
X = df.drop(['Y'], axis=1).values

#parameter setting
X = normalize(X, 150)

print('X zero')
print(X[0])

X_u = X[P==1]
X_n = X[P==0]

print('original emd distance:')
print(cal_emd_resamp(X_u, X_n, 50, 10))
print('original emd distance without P:')

print(cal_emd_resamp(X_u[:,:-1], X_n[:,:-1], 50, 10))
print('original positive group distance without P:')
print(cal_emd_resamp(X[:,:-1][(y==1) & (P==0)], X[:,:-1][(y==1) & (P==1)], 50, 10))
print('original negative group distance without P:')
print(cal_emd_resamp(X[:,:-1][(y==0) & (P==0)], X[:,:-1][(y==0) & (P==1)], 50, 10))

X = torch.tensor(X).float()

n_dim = 30
batch_size = 2000
n_iter = 20
C=0.1
alpha = 1000
k_nbrs= 1

n_test = 2
results = {}
y_hats = {}

preds = {}
reps = {}
for k in range(n_test):
    results_this, y_test_this, reps_this = test_in_one(n_dim=n_dim,
                     batch_size=batch_size,
                     n_iter=n_iter,
                     C=C,
                     alpha=alpha,
                    compute_emd=False,
                    k_nbrs=k_nbrs,
                    emd_method=lambda x,y: cal_emd_resamp(x, y, 50, 10))

    if k == 0:
        results = results_this
        for model in results:
            results[model] = np.array(results_this[model])/ n_test
            preds[model] = y_test_this[model] / n_test
            if torch.is_tensor(reps_this[model]):
                reps[model] = reps_this[model] / n_test
            else:
                reps[model] = None
    else:
        for model in results:
            results[model] += np.array(results_this[model]) / n_test
            preds[model] += y_test_this[model] / n_test
            if torch.is_tensor(reps_this[model]):
                reps[model] += reps_this[model] / n_test
            else:
                reps[model] = None

for key, val in preds.items():
    save_predictions(y, preds[key], reps[key], key)


# TODO combine with csv
print('Predicted y saved to compas_y_pred.csv')
print('{0:40}: {1}'.format('method', ' '.join(['ks', 'recall', 'precision', 'f1','stat','emd','cons', 'stat_abs'])))
for key, val in results.items():
    print('{0:40}: {1}'.format(key, ' '.join([str(np.round(x,3)) for x in val]).ljust(35)))
