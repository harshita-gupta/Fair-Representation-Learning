
# coding: utf-8

# In[1]:



import torch
import torch.nn as nzn
import torch.optim as optim
import torch.distributions as D

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
    #score1 = np.mean(scores[P==1])
    #score0 = np.mean(scores[P==0])
    #return 1.0*max(score1,score0)/min(score1,score0)
    return np.abs(np.mean(scores[P==0]) - np.mean(scores[P==1]))


# In[3]:


def test_in_one(n_dim, batch_size, n_iter, C, alpha,compute_emd=True, k_nbrs = 3, emd_method=emd_samples):
    global X, P, y
    # AE.
    model_ae = FairRep(len(X[0]), n_dim)
    model_ae.cuda()
    X = torch.tensor(X).float().cuda()
    P = torch.tensor(P).long().cuda()
    train_rep(model_ae, 0.01, X, P, n_iter, 10, batch_size, alpha = 0, C_reg=0, compute_emd=compute_emd, adv=False, verbose=True)
    # AE_P.
    model_ae_P = FairRep(len(X[0])-1, n_dim-1)
    model_ae_P.cuda()
    X = torch.tensor(X).float().cuda()
    P = torch.tensor(P).long().cuda()
    train_rep(model_ae_P, 0.01, X[:, :-1], P, n_iter, 10, batch_size, alpha = 0, C_reg=0, compute_emd=compute_emd, adv=False, verbose=True)
    # NFR.
    model_nfr = FairRep(len(X[0]), n_dim)
    model_nfr.cuda()
    X = torch.tensor(X).float().cuda()
    P = torch.tensor(P).long().cuda()
    train_rep(model_nfr, 0.01, X, P, n_iter, 10, batch_size, alpha = alpha, C_reg=0, compute_emd=compute_emd, adv=True, verbose=True)
    results={}

    print('begin testing.')
    X_ori_np = X.data.cpu().numpy()
    # Original.
    data_train, data_test = split_data_np((X.data.cpu().numpy(),P.data.cpu().numpy(),y), 0.7)
    X_train, P_train, y_train = data_train
    X_test, P_test, y_test = data_test
    print('logistic regresison on the original...')
    lin_model = LogisticRegression(C=C, solver='sag', max_iter=2000)
    lin_model.fit(X_train, y_train)
    #print(lin_model.coef_.shape)
    #int(X_train.shape)

    y_test_scores = sigmoid((X_test.dot(lin_model.coef_.T) + lin_model.intercept_).flatten())
    print('logistic regresison evaluation...')
    performance = list(evaluate_performance_sim(y_test, y_test_scores, P_test))
    print('calculating emd...')
    performance.append(emd_method(X_n, X_u))
    print('calculating consistency...')
    performance.append(get_consistency(X.data.cpu().numpy(), lin_model, n_neighbors=k_nbrs))
    print('calculating stat diff...')
    performance.append(stat_diff(X.data.cpu().numpy(), P, lin_model))
    results['Original'] = performance
    # Original-P.
    data_train, data_test = split_data_np((X[:, :-1].data.cpu().numpy(),P.data.cpu().numpy(),y), 0.7)
    X_train, P_train, y_train = data_train
    X_test, P_test, y_test = data_test
    print('logistic regresison on the original-P')
    lin_model = LogisticRegression(C=C, solver='sag', max_iter=2000)
    lin_model.fit(X_train, y_train)

    y_test_scores = sigmoid((X_test.dot(lin_model.coef_.T) + lin_model.intercept_).flatten())
    print('logistic regresison evaluation...')
    performance = list(evaluate_performance_sim(y_test, y_test_scores, P_test))
    print('calculating emd...')
    performance.append(emd_method(X_n[:,:-1], X_u[:,:-1]))
    print('calculating consistency...')
    performance.append(get_consistency(X[:,:-1].data.cpu().numpy(), lin_model,  n_neighbors=k_nbrs))
    print('calculating stat diff...')
    performance.append(stat_diff(X[:,:-1].data.cpu().numpy(), P, lin_model))
    results['Original-P'] = (performance)
    U_0 = model_ae.encoder(X[P==0]).data
    U_1 = model_ae.encoder(X[P==1]).data
    U = model_ae.encoder(X).data
    print('ae emd afterwards: ' + str(emd_method(U_0, U_1)))
    U_np = U.cpu().numpy()
    data_train, data_test = split_data_np((U_np,P.data.cpu().numpy(),y), 0.7)
    X_train, P_train, y_train = data_train
    X_test, P_test, y_test = data_test

    print('logistic regresison on AE...')
    lin_model = LogisticRegression(C=C, solver='sag', max_iter=2000)
    lin_model.fit(X_train, y_train)

    y_test_scores = sigmoid((X_test.dot(lin_model.coef_.T) + lin_model.intercept_).flatten())
    print('logistic regresison evaluation...')
    performance = list(evaluate_performance_sim(y_test, y_test_scores, P_test))
    print('calculating emd...')
    performance.append(emd_method(U_0, U_1))
    print('calculating consistency...')
    performance.append(get_consistency(U_np, lin_model, n_neighbors=k_nbrs, based_on=X_ori_np))
    print('calculating stat diff...')
    performance.append(stat_diff(X_test, P_test, lin_model))
    results['AE'] = (performance)


    U_0 = model_ae_P.encoder(X[:,:-1][P==0]).data
    U_1 = model_ae_P.encoder(X[:,:-1][P==1]).data
    U = model_ae_P.encoder(X[:,:-1]).data
    print('ae-p emd afterwards: ' + str(emd_method(U_0, U_1)))
    U_np = U.cpu().numpy()
    data_train, data_test = split_data_np((U_np,P.data.cpu().numpy(),y), 0.7)
    X_train, P_train, y_train = data_train
    X_test, P_test, y_test = data_test

    print('logistic regresison on AE-P...')
    lin_model = LogisticRegression(C=C, solver='sag', max_iter=2000)
    lin_model.fit(X_train, y_train)

    y_test_scores = sigmoid((X_test.dot(lin_model.coef_.T) + lin_model.intercept_).flatten())
    print('logistic regresison evaluation...')
    performance = list(evaluate_performance_sim(y_test, y_test_scores, P_test))
    print('calculating emd...')
    performance.append(emd_method(U_0, U_1))
    print('calculating consistency...')
    performance.append(get_consistency(U_np, lin_model,  n_neighbors=k_nbrs, based_on=X_ori_np))
    print('calculating stat diff...')
    performance.append(stat_diff(X_test, P_test, lin_model))
    results['AE_P'] = (performance)

    U_0 = model_nfr.encoder(X[P==0]).data
    U_1 = model_nfr.encoder(X[P==1]).data
    U = model_nfr.encoder(X).data
    print('nfr emd afterwards: ' + str(emd_method(U_0, U_1)))

    U_np = U.cpu().numpy()
    data_train, data_test = split_data_np((U_np,P.data.cpu().numpy(),y), 0.7)
    X_train, P_train, y_train = data_train
    X_test, P_test, y_test = data_test
    print('logistic regresison on NFR...')
    lin_model = LogisticRegression(C=C, solver='sag', max_iter=2000)
    lin_model.fit(X_train, y_train)

    y_test_scores = sigmoid((X_test.dot(lin_model.coef_.T) + lin_model.intercept_).flatten())
    print('logistic regresison evaluation...')
    performance = list(evaluate_performance_sim(y_test, y_test_scores, P_test))
    print('calculating emd...')
    performance.append(emd_method(U_0, U_1))
    print('calculating consistency...')
    performance.append(get_consistency(U_np, lin_model, n_neighbors=k_nbrs, based_on=X_ori_np))
    print('calculating stat diff...')
    performance.append(stat_diff(X_test, P_test, lin_model))
    results['NFR'] = (performance)

    return results


# In[4]:



# two batch of samples: one normal(0,1), and one uniform(0,1).
with open('data/german.numeric.processed') as f:
    data_raw = np.array([list(map(float, x)) for x in map(lambda x: x.split(), f)])
    data_raw = np.array(data_raw)
np.random.shuffle(data_raw)
P = data_raw[:, -2]
y = data_raw[:, -1]
X = data_raw[:, :-1]


#parameter setting
X = normalize(X, 150)

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

# In[9]:


print(X.shape)


# In[ ]:


n_dim = 30
batch_size = 2000
n_iter = 20
C=0.1
alpha = 1000
k_nbrs= 1

n_test = 2
results = {}
for k in range(n_test):
    results_this = test_in_one(n_dim=n_dim,
                     batch_size=batch_size,
                     n_iter=n_iter,
                     C=C,
                     alpha=alpha,
                    compute_emd=False,
                    k_nbrs=k_nbrs,
                    emd_method=lambda x,y: cal_emd_resamp(x, y, 50, 10))
    #print(results_this)
    if k == 0:
        results = results_this
        for model in results:
            results[model] = np.array(results_this[model])/n_test
    else:
        for model in results:
            results[model] += np.array(results_this[model]) / n_test

print('{0:40}: {1}'.format('method', ' '.join(['ks', 'recall', 'precision', 'f1','stat','emd','cons', 'stat_abs'])))
for key, val in results.items():
    print('{0:40}: {1}'.format(key, ' '.join([str(np.round(x,3)) for x in val]).ljust(35)))

