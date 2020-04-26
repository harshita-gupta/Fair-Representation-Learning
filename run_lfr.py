# k = number of propotypes
import pickle
import os
import numpy as np
import csv
import scipy.optimize as optim
from helpers_lfr import *
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import sys
from helpers import update_progress, normalize, total_correlation


#k = int(sys.argv[2])
#dataset = sys.argv[0]
#norm_ = int(sys.argv[1])

k = 10
dataset = 'german'
norm_ = 20

# k=10
# dataset = 'gps'
# norm_=20

with open('data/'+dataset+'.numeric.processed') as f:
    dat = np.array([list(map(float, x)) for x in map(lambda x: x.split(), f)])

# print dat.shape
# print 'processing '+dataset+' dataset...'


print('finished reading data')
data = np.array(dat)

y = np.array(data[:,-1]).flatten()
data = data[:,:-1]
sensitive = data[:,-1]

print('total instances: '+ str(data.shape))
# data = preprocessing.scale(data)
data = normalize(data,norm_)


data = data[:,:-1]
#remove sensitive


# separate data into sensitive and nonsensitive

sensitive_idx = np.array(np.where(sensitive == 1))[0].flatten()
nonsensitive_idx = np.array(np.where(sensitive != 1))[0].flatten()

data_sensitive = data[sensitive_idx,:]
data_nonsensitive = data[nonsensitive_idx,:]

y_sensitive = y[sensitive_idx]
y_nonsensitive = y[nonsensitive_idx]

print('sensitive instances: '+str(data_sensitive.shape))
print('nonsensitive instances: '+str(data_nonsensitive.shape))

import random
random.seed(1)

train_set_ratio = 0.7
print('\n')
print('shuffle sensitive data...')
random.shuffle(list(zip(data_sensitive, y_sensitive)))

training_sensitive = data_sensitive[:int(data_sensitive.shape[0]*train_set_ratio)]
ytrain_sensitive = y_sensitive[:int(data_sensitive.shape[0]*train_set_ratio)]

# random.shuffle(zip(data_nonsensitive,y_nonsensitive))

test_sensitive = data_sensitive[int(data_sensitive.shape[0]*train_set_ratio):]
ytest_sensitive = y_sensitive[int(data_sensitive.shape[0]*train_set_ratio):]

print('Sensitive training data: '+str(training_sensitive.shape))
print('Sensitive test data: '+str(test_sensitive.shape))
# training_sensitive = random.sample()
# idx=indices[2]

print('shuffle nonsensitive data...')
random.shuffle(list(zip(data_nonsensitive,y_nonsensitive)))

training_nonsensitive = data_nonsensitive[:int(data_nonsensitive.shape[0]*train_set_ratio)]
ytrain_nonsensitive = y_nonsensitive[:int(data_nonsensitive.shape[0]*train_set_ratio)]

# idx2=indices[3]
test_nonsensitive = data_nonsensitive[int(data_nonsensitive.shape[0]*train_set_ratio):]
ytest_nonsensitive = y_nonsensitive[int(data_nonsensitive.shape[0]*train_set_ratio):]

print('Nonsensitive training data: '+str(training_nonsensitive.shape))
print('Nonsensitive test data: '+str(test_nonsensitive.shape))
print('\n')

training = np.concatenate((training_sensitive, training_nonsensitive))
ytrain = np.concatenate((ytrain_sensitive, ytrain_nonsensitive))

test = np.concatenate((test_sensitive, test_nonsensitive))
ytest = np.concatenate((ytest_sensitive, ytest_nonsensitive))

print('Whole training dataset: '+str(training.shape))
print('Whole test dataset: '+str(test.shape))
print('\n')


print('Initialize w and v...\n')
rez = np.random.uniform(size=data.shape[1] * 2 + k + data.shape[1] * k)


yhat_sensitive, yhat_nonsensitive, M_nk_sensitive, M_nk_nonsensitive = LFR(rez,
                                                                           training_sensitive,
                                                                           training_nonsensitive,
                                                                           ytrain_sensitive,
                                                                           ytrain_nonsensitive,
                                                                           k, 1e-4, 0.1, 1000,
                                                                           results=1)
# print yhat_sensitive.shape
# print yhat_nonsensitive.shape
# print M_nk_sensitive.shape
# print M_nk_nonsensitive.shape


test_yhat_sensitive, test_yhat_nonsensitive, test_M_nk_sensitive, test_M_nk_nonsensitive = LFR(rez,
                                                                           test_sensitive,
                                                                           test_nonsensitive,
                                                                           ytest_sensitive,
                                                                           ytest_nonsensitive,
                                                                           k, 1e-4, 0.1, 1000,
                                                                           results = 1)

# print test_yhat_sensitive.shape
# print test_yhat_nonsensitive.shape
# print test_M_nk_sensitive.shape
# print test_M_nk_nonsensitive.shape


# only assign bounds to w
bnd = []
for i, k2 in enumerate(rez):
    if i < data.shape[1] * 2 or i >= data.shape[1] * 2 + k:
        bnd.append((None, None))
    else:
        bnd.append((0, 1))

print('begin optimizing...')

rez = optim.fmin_l_bfgs_b(LFR, x0=rez, epsilon=1e-5,
                          args=(training_sensitive, training_nonsensitive,
                                ytrain_sensitive, ytrain_nonsensitive, k, 1e-4,
                                0.1, 1000, 0),
                          bounds = bnd, approx_grad=True, maxfun=150000,
                          maxiter=150000, iprint=-1)

w = rez[0][data.shape[1]*2:data.shape[1]*2+k]

v = rez[0][data.shape[1]*2+k:].reshape(k,data.shape[1])

# w = rez[data.shape[1]*2:data.shape[1]*2+k]

# v = rez[data.shape[1]*2+k:].reshape(k,data.shape[1])

from dumb_containers import evaluate_performance, evaluate_performance_sim

ytest_sensitive_pred = test_M_nk_sensitive.dot(np.expand_dims(w,axis=1))
# print ytest_sensitive_pred.shape

# evaluate_performance_sim(ytest_sensitive, ytest_sensitive_pred)

ytest_nonsensitive_pred = test_M_nk_nonsensitive.dot(np.expand_dims(w,axis=1))
# print ytest_nonsensitive_pred.shape

# evaluate_performance_sim(ytest_nonsensitive, ytest_nonsensitive_pred)
ytest_sensitive = ytest_sensitive.flatten()
ytest_sensitive = list(ytest_sensitive)

ytest_nonsensitive = ytest_nonsensitive.flatten()
ytest_nonsensitive = list(ytest_nonsensitive)

target = ytest_sensitive + ytest_nonsensitive
# len(target)

ytest_sensitive_pred = ytest_sensitive_pred.flatten()
ytest_sensitive_pred = list(ytest_sensitive_pred)

ytest_nonsensitive_pred = ytest_nonsensitive_pred.flatten()
ytest_nonsensitive_pred = list(ytest_nonsensitive_pred)

pred = ytest_sensitive_pred + ytest_nonsensitive_pred
# len(pred)

# P_label = np.ones()
P_sensitive = list(np.ones(len(ytest_sensitive)))
P_nonsensitive = list(np.zeros(len(ytest_nonsensitive)))
P = P_sensitive+P_nonsensitive
# len(P)

KS, recall, precision, f1, parity_dev, parity_sub, event_rate = evaluate_performance_sim(np.array(target), np.array(pred), np.array(P), more_eva =1)

result = [KS, recall, precision, f1, parity_dev, parity_sub]
# print result
print('threshold: '+str(np.round(event_rate,4)))
from pyemd import emd_samples

def cal_emd_resamp(A,B,n_samp,times):
    emds = []
    for t in range(times):
        idx_a = np.random.choice(len(A), n_samp)
        idx_b = np.random.choice(len(B), n_samp)
        emds.append(emd_samples(A[idx_a],B[idx_b]))
    return np.mean(emds)

X_sen_hat = test_M_nk_sensitive.dot(v)
X_nonsen_hat = test_M_nk_nonsensitive.dot(v)
# print('emd distance:')
emd = cal_emd_resamp(X_sen_hat, X_nonsen_hat, 1000, 10)
# print np.round(emd,3)


result.append(np.round(emd,4))
# print result



# consistency
from sklearn.neighbors import NearestNeighbors

all_x = np.vstack((X_sen_hat, X_nonsen_hat))
K=4

nbrs = NearestNeighbors(n_neighbors=K+1).fit(all_x)
distances, indices = nbrs.kneighbors(all_x)

knn_y = np.array([i>event_rate for i in pred])
knn_y = knn_y+1-1

pred_nbr = np.array([knn_y[i] for i in indices.reshape(1,indices.shape[0]*indices.shape[1])]).reshape(indices.shape[0],indices.shape[1])
con_clf = 1-sum(abs(pred_nbr[:,0]-np.sum(pred_nbr[:,1:],axis=1)/K)/len(pred_nbr))

result.append(np.round(con_clf,4))

# mse
test_X = np.vstack((X_sen_hat,X_nonsen_hat))
raw_X = np.vstack((test_sensitive,test_nonsensitive))

mse = np.mean(np.power(test_X-raw_X, 2))

result.append(np.round(mse,4))

print(['ks', ' recall', ' prec', ' f1','  par_div','par_sub','emd','consis','mse'])
print(result)
