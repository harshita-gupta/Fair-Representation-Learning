
import time
import sys
import pandas as pd
import copy
import numpy as np
from pyemd import emd_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import torch

def prettytime(seconds):
    return seconds/3600, seconds/60%60, seconds%60

def update_progress(i, total, start_time, text=''):
    now = time.time()
    used = prettytime(now-start_time)
    eta = prettytime((now-start_time) / (i+1) * (total-i-1))
    output = ("\r%.2f%%, " % (100.0 * (i+1)/total) +
                 "%d/%d processed, " % (i+1, total) + text +
                 "time used: %02d:%02d:%02d, eta: %02d:%02d:%02d" %
                (used[0], used[1], used[2],
                eta[0], eta[1], eta[2]))
    sys.stdout.write(output)
    sys.stdout.flush()
    if i == total-1:
        print('')


def isfloat(string):
    try:
        x = float(string)
    except ValueError:
        return False
    return True


def load_adult_data(url):
    data_train = pd.read_csv(url)
    columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-weak','native-country','income']
    first_row = data_train.columns
    data_train.columns = columns
    data_train.loc[-1] = first_row
    data_train.index += 1
    data_train = data_train.sort_index()
    # data_train = data_train.drop('fnlwgt',axis=1)
    for c in data_train.columns:
        if data_train[c].dtype == 'object':
            if isfloat(data_train[c][0]):
                data_train[c] = data_train[c].astype(float)

    #data_train = pd.get_dummies(data_train, drop_first=True)

    return data_train


def get_dummies_map(data):
    dummies = {}
    for c in data:
        if data[c].dtype == 'object':
            objs = sorted(set(data[c]))
            mapping = {objs[i]:i for i in range(len(objs))}
            dummies[c] = mapping

    return dummies


def convert_dummies(data, dummies):
    data_new = copy.deepcopy(data)
    for c in dummies:
        data_new = data_new.replace(dummies[c])
    return data_new


def normalize(data, n_unique):
    data_new = np.array(copy.deepcopy(data)).T
    for i, col in enumerate(data_new):
        n_uni = len(set(data_new[i, :]))
        if n_uni >= n_unique:
            tmp = data_new[i, :]
            data_new[i, :] = (tmp - np.mean(tmp)) / np.std(tmp)
    return data_new.T


def total_correlation(X, Y):
    X = normalize(X,0)
    N = len(X)
    #Y_norm = normalize(Y, 10)
    S_XX = 1.0 * X.T.dot(X) / N
    S_YX = 1.0 * Y.T.dot(X) / N
    S_XY = 1.0 * X.T.dot(Y) / N
    S_XX_inv = 1.0 * np.linalg.inv(S_XX)
    S_YY = 1.0 * Y.T.dot(Y) / N

    R_sq = S_YX.dot(S_XX_inv).dot(S_XY) / S_YY
    return np.sqrt(R_sq)


def cal_emd_resamp(A,B,n_samp,times):
    emds = []
    for t in range(times):
        idx_a = np.random.choice(len(A), n_samp)
        idx_b = np.random.choice(len(B), n_samp)
        emds.append(emd_samples(A[idx_a],B[idx_b], bins=2))
    return np.mean(emds)

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


def make_cal_plot(X, y, P, model, model_name):
    """
    Saves a calibration plot for the given model.
    See https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html

    Parameters:
    X: features used to make predictions for model.
    y: true labels.
    P: indicator for protected class membership.
    model: trained LogisticRegression model.
    model_name: name of model.

    Returns:
    Saves plot to 'results/model_name.png'.
    """
    scores = sigmoid(X.dot(model.coef_.T) + model.intercept_)
    fop, mpv = calibration_curve(y, scores, n_bins=10, normalize=True)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline')
    # plot model reliability
    plt.plot(list(mpv), list(fop), marker='.', label='All data')

    # protected class
    scores_p = scores[P == 1]
    y_p = y[P == 1]
    scores_np = scores[P == 0]
    y_np = y[P == 0]
    fop, mpv = calibration_curve(y_p, scores_p, n_bins=10, normalize=True)
    plt.plot(list(mpv), list(fop), marker=',', label='P = 1')

    fop, mpv = calibration_curve(y_np, scores_np, n_bins=10, normalize=True)
    plt.plot(list(mpv), list(fop), marker='1', label='P = 0')
    plt.title(model_name + ' calibration plot')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.legend()
    plt.show(block=False)
    plt.savefig('results/' + model_name + '.png')
    plt.clf()

def stat_diff(X, P, model):
    scores = sigmoid(X.dot(model.coef_.T) + model.intercept_)
    return np.abs(np.mean(scores[P==0]) - np.mean(scores[P==1]))

def equal_odds(X, y, P, model):
    X_p = X[P == 1]
    y_p = y[P == 1]
    X_np = X[P == 0]
    y_np = y[P == 0]

    # given y = 1, find difference in predicted scores

    i_p_pos = np.argwhere(y_p == 1)
    X_p_pos = np.take(X_p, i_p_pos, axis=0)
    scores_p = sigmoid(X_p_pos.dot(model.coef_.T) + model.intercept_)

    i_np_pos = np.argwhere(y_np == 1)
    X_np_pos = np.take(X_np, i_np_pos, axis=0)
    scores_np = sigmoid(X_np_pos.dot(model.coef_.T) + model.intercept_)

    diff_pos = np.abs(np.mean(scores_p) - np.mean(scores_np))

    # given y = 0, find difference in predicted scores

    i_p_neg = np.argwhere(y_p == 0)
    X_p_neg = np.take(X_p, i_p_neg, axis=0)
    scores_p = sigmoid(X_p_neg.dot(model.coef_.T) + model.intercept_)

    i_np_neg = np.argwhere(y_np == 0)
    X_np_neg = np.take(X_np, i_np_neg, axis=0)
    scores_np = sigmoid(X_np_neg.dot(model.coef_.T) + model.intercept_)

    diff_neg = np.abs(np.mean(scores_p) - np.mean(scores_np))

    return diff_pos + diff_neg

def save_predictions(df, X, y, y_hat, reps, model_name):
    # make CSV dataframe to store predicted scores
    y_hat = y_hat.reshape(len(X), 1)
    y = y.reshape(len(X), 1)

    data_yhat = np.concatenate((X, y, y_hat), axis=1)
    cols = list(df.columns)
    cols.append('y_hat')
    pred_df = pd.DataFrame(data = data_yhat, columns = cols)
    pred_df.to_csv('results/preds_' + model_name + '.csv')

    if torch.is_tensor(reps):
        reps_np = reps.numpy()
        data_reps = np.concatenate((reps_np, y, y_hat), axis=1)
        num_cols = reps_np.shape[1]
        cols = []
        for i in range(num_cols):
            cols.append('repr_' + str(i))
        cols += ['y', 'y_hat']
        repr_df = pd.DataFrame(data = data_reps, columns = cols)
        repr_df.to_csv('results/representation_' + model_name + '.csv')
