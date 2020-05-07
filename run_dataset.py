import torch
import torch.nn as nzn
import torch.optim as optim
import torch.distributions as D
import os
import sys

import numpy as np
from pyemd import emd_samples

from model import FairRep
from helpers import update_progress, normalize, total_correlation, cal_emd_resamp, save_decision_boundary_plot
from helpers import split_data_np, get_consistency, stat_diff, equal_odds, sigmoid, make_cal_plot, save_predictions
import time
import sys
from train import train_rep
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from dumb_containers import split_data, evaluate_performance_sim
import matplotlib.pyplot as plt
import pandas as pd
from math import log10
import argparse

run_alpha_cv = False

np.random.seed(1)

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

def run_nfr_cv(n_dim, batch_size, C, alpha, emd_method = emd_samples):
    global X, P, y, df, X_test

    reps = {}

    # X_no_p = df.drop([y_name, prot], axis=1).values

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

    # NFR.
    model_nfr = FairRep(len(X[0]), n_dim)
    X = torch.tensor(X).float()
    P = torch.tensor(P).long()
    return train_rep(model_nfr, 0.01, X, P, n_iter, c_iter=10, batch_size=batch_size, alpha = alpha, C_reg=C)

def test_in_one(n_dim, batch_size, n_iter, C, alpha,compute_emd=True, k_nbrs = 3, emd_method=emd_samples):
    global X, P, y, df, X_test, X_notp

    reps = {}

    # X_no_p = df.drop([y_name, prot], axis=1).values

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
    train_rep(model_ae_P, 0.01,X_notp , P, n_iter, 10, batch_size, alpha = 0, C_reg=0, compute_emd=compute_emd, adv=False, verbose=True)

    # NFR.
    model_name = 'compas_Original'
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
    performance.append(equal_odds(X.data.cpu().numpy(), y, P, lin_model))
    make_cal_plot(X.data.cpu().numpy(), y, P, lin_model, model_name)
    save_decision_boundary_plot(
        np.concatenate((X_train, X_test)),
        np.concatenate((y_train, y_test)), np.concatenate((P_train, P_test)), model_name)

    results[model_name] = performance

    # Original-P.
    model_name = 'compas_Original-P'
    print('logistic regression on the original-P')
    lin_model, y_test_scores, performance = get_model_preds(X_train_no_p, y_train, P_train, X_test_no_p, y_test, P_test, model_name)
    y_hats[model_name] = get_preds_on_full_dataset(X[:, :-1], lin_model)
    reps[model_name] = None

    performance.append(emd_method(X_n[:,:-1], X_u[:,:-1]))
    print('calculating consistency...')
    performance.append(get_consistency(X[:,:-1].data.cpu().numpy(), lin_model,  n_neighbors=k_nbrs))
    print('calculating stat diff...')
    performance.append(stat_diff(X[:,:-1].data.cpu().numpy(), P, lin_model))
    performance.append(equal_odds(X[:,:-1].data.cpu().numpy(), y, P, lin_model))
    make_cal_plot(X[:,:-1].data.cpu().numpy(), y, P, lin_model, model_name)
    save_decision_boundary_plot(
        np.concatenate((X_train_no_p, X_test_no_p)),
        np.concatenate((y_train, y_test)), np.concatenate((P_train, P_test)), model_name)

    results[model_name] = performance




    # use encoder
    model_name = 'compas_AE'

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

    def calc_perf(y_test, y_test_scores, P_test, U, U_0, U_1, U_np, lin_model, X_test, model_name):
        print('logistic regression evaluation...')
        performance = list(evaluate_performance_sim(y_test, y_test_scores, P_test))
        print('calculating emd...')
        performance.append(emd_method(U_0, U_1))
        print('calculating consistency...')
        performance.append(get_consistency(U_np, lin_model, n_neighbors=k_nbrs, based_on=X_ori_np))
        print('calculating stat diff...')
        performance.append(stat_diff(X_test, P_test, lin_model))
        print('calculating equal odds...')
        performance.append(equal_odds(X_test, y_test, P_test, lin_model))
        make_cal_plot(X_test, y_test, P_test, lin_model, model_name)
        return performance

    performance = calc_perf(y_test, y_test_scores, P_test, U, U_0, U_1, U_np, lin_model, X_test, model_name)
    save_decision_boundary_plot(
        np.concatenate((X_train, X_test)),
        np.concatenate((y_train, y_test)), np.concatenate((P_train, P_test)), model_name)
    results[model_name] = (performance)



    # AE minus P
    model_name = 'compas_AE_P'
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
    save_decision_boundary_plot(
        np.concatenate((X_train, X_test)),
        np.concatenate((y_train, y_test)), np.concatenate((P_train, P_test)), model_name)
    

    performance = calc_perf(y_test, y_test_scores, P_test, U, U_0, U_1, U_np, lin_model, X_test, model_name)
    results[model_name] = (performance)

    model_name = 'compas_NFR'
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
    save_decision_boundary_plot(
        np.concatenate((X_train, X_test)),
        np.concatenate((y_train, y_test)), np.concatenate((P_train, P_test)), model_name)

    performance = calc_perf(y_test, y_test_scores, P_test, U, U_0, U_1, U_np, lin_model, X_test, model_name)
    results[model_name] = (performance)

    return results, y_hats, reps



# two batch of samples: one normal(0,1), and one uniform(0,1).
# with open('data/german.numeric.processed') as f:
    # data_raw = np.array([list(map(float, x)) for x in map(lambda x: x.split(), f)])
    # print('raw data')
    # print(data_raw)
    # data_raw = np.array(data_raw)

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store', type=str)
parser.add_argument('--alpha', action='store', type=float)
parser.add_argument('--ndim', action='store', type=int)
parser.add_argument('--protected', action='store', type=str)
parser.add_argument('--y', action='store', type=str)
parser.add_argument('--bins', action='store', type=int)
parser.add_argument('--n_iter', action='store', type=int)
parser.add_argument('--c', action='store', type=int)


args = parser.parse_args()

# default options
data_name = 'compas_new.csv'
n_dim = 30
alpha = 1000
prot = 'P'
y_name = 'Y'
bins = 2
n_iter = 20
C=0.1

# if alternatives are entered in the command line, use those instead
if args.data:
    data_name = args.data
if args.ndim:
    n_dim = args.ndim
if args.alpha:
    alpha = args.alpha
if args.protected:
    prot = args.protected
if args.y:
    y_name = args.y
if args.bins:
    bins = args.bins
if args.n_iter:
    n_iter = args.n_iter
if args.c:
    C = args.c

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', data_name)
#filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'german_clean.csv')
try:
    df = pd.read_csv(filepath)
except IOError as err:
    print('IO error')

print('num COLs:')
print(len(list(df.columns)))

P = df[prot].values
y = df[y_name].values

# X contains protected class P
X = df.drop([y_name], axis=1).values

X_notp =  df.drop([y_name, prot], axis=1).values
#parameter setting
X = normalize(X, 150)
X_notp = normalize(X_notp, 150)

print('X zero')
print(X[0])

X_u = X[P==1]
X_n = X[P==0]


# print("compare the unprotected normalized data:", sum(X_n[:,:-1] != X_notp[P==0]) )


print('original emd distance:')
print(cal_emd_resamp(X_u, X_n, 50, 10, bins))
print('original emd distance without P:')

print(cal_emd_resamp(X_notp[P==1], X_notp[P==0], 50, 10, bins))
print('original positive group distance without P:')
print(cal_emd_resamp(X_notp[(y==1) & (P==0)], X_notp[(y==1) & (P==1)], 50, 10, bins))
print('original negative group distance without P:')
print(cal_emd_resamp(X_notp[(y==0) & (P==0)], X_notp[(y==0) & (P==1)], 50, 10, bins))

X = torch.tensor(X).float()

batch_size = 2000
k_nbrs= 1

n_test = 2
results = {}
y_hats = {}

preds = {}
reps = {}

if run_alpha_cv:
    dataset = 'compas'
    # cross-validation on alpha
    alph_results = []
    for alph in [10**2, 10**3, 10**4, 10**6, 10**8, 10**10]:
        mse, wdist = run_nfr_cv(n_dim=n_dim, batch_size=batch_size, C=1., alpha=alph, emd_method = emd_samples)
        alph_results.append({'alpha': alph, 'mse': mse.detach().numpy(), 'wdist': wdist.detach().numpy()})
    alph_df = pd.DataFrame(alph_results)
    alph_df.to_csv('results/' + dataset + 'alpha_cv.csv')
    print('saved, exiting')

    # make plot
    plt.plot(np.log10(alph_df['alpha']), alph_df['wdist'], color='red')
    plt.xlabel('Alpha')
    plt.ylabel('Wasserstein Distance')
    plt.show(block=False)
    plt.title(dataset + ': Alpha vs. Wasserstein Distance')
    plt.savefig('results/' + dataset + '_alpha_wdist.png')
    plt.clf()

    plt.plot(np.log10(alph_df['alpha']), alph_df['mse'], color='black', linestyle='dashed', label='MSE')
    plt.xlabel('Alpha')
    plt.ylabel('MSE')
    plt.show(block=False)
    plt.title(dataset + ': Alpha vs. MSE')
    plt.savefig('results/' + dataset + '_alpha_mse.png')
    plt.clf()



    # cross-validation on n_dim
    dim_results = []
    dim_range = np.linspace(1, len(X[0]), 10)
    for dim in dim_range:
        d_i = int(dim)
        mse, wdist = run_nfr_cv(n_dim=d_i, batch_size=batch_size, C=1., alpha=alpha, emd_method = emd_samples)
        dim_results.append({'n_dim': d_i, 'mse': mse.detach().numpy(), 'wdist': wdist.detach().numpy()})
    dim_df = pd.DataFrame(dim_results)
    dim_df.to_csv(dataset + '_dim_cv.csv')
    print('results/saved, exiting')

    plt.plot(dim_df['n_dim'], dim_df['wdist'], color='red')
    plt.xlabel('n_dim')
    plt.ylabel('Wasserstein Distance')
    plt.show(block=False)
    plt.title(dataset + ': n_dim vs. Wasserstein Distance')
    plt.savefig('results/' + dataset + '_dim_wdist.png')
    plt.clf()

    plt.plot(dim_df['n_dim'], dim_df['mse'], color='black', linestyle='dashed', label='MSE')
    plt.xlabel('n_dim')
    plt.ylabel('MSE')
    plt.show(block=False)
    plt.title(dataset + ': n_dim vs. MSE')
    plt.savefig('results/' + dataset + '_dim_mse.png')
    plt.clf()




else:
    for k in range(n_test):
        results_this, y_test_this, reps_this = test_in_one(n_dim=n_dim,
                         batch_size=batch_size,
                         n_iter=n_iter,
                         C=C,
                         alpha=alpha,
                        compute_emd=False,
                        k_nbrs=k_nbrs,
                        emd_method=lambda x,y: cal_emd_resamp(x, y, 50, 10, bins))

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
        save_predictions(df, X, y, preds[key], reps[key], key)


    # TODO combine with csv
    print('Predictions saved.')
    print('{0:40}: {1}'.format('method', ' '.join(['ks', 'recall', 'precision', 'f1','stat','emd','cons', 'stat_abs', 'eq_odds'])))
    for key, val in results.items():
        print('{0:40}: {1}'.format(key, ' '.join([str(np.round(x,3)) for x in val]).ljust(35)))

    print('Complete.')
