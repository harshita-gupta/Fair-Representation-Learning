
import time
import sys
import pandas as pd
import copy
import numpy as np
from pyemd import emd_samples


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
