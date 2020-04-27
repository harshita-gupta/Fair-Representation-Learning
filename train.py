
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

import numpy as np
from pyemd import emd_samples

from model import FairRep
from helpers import update_progress, normalize
from dumb_containers import evaluate_performance_sim
import time



def train_rep(model, lr, X, P, n_iter, c_iter, batch_size,
              alpha = 10, C_reg = 1,
              compute_emd=False, adv=True, verbose=False):
    """
    Train the fair representation using autoencoder provided by the user.
    Parameters:
        model: the Pytorch model of the autoencoder. The model should have two members, model.encoder, and the model.decoder.

    Parameters:
        lr: learning rate.
        X: the input features.
        P: the protected attribute.
        n_iter: number of iterations.
        c_iter: the number of iteration to trian the critic inside each training iteration.
        batch_size: batch size.
        alpha: the weight of the fairness contraint. Larger means more penalize on the violation of fairness.
        C_reg: the penalization coefficient of the regularization of the encoder.
        compute_emd: whether the EMD distance is calculated for each iteration. It may slow the training process significantly.
        adv: if the model is trained adversarially, i.e. fairly. Setting it false will result in training a normal autoencoder.
        verbose: if the training process is verbosely printed.
    """
    time_s = time.time()
    X_0 = X[P == 0]
    X_1 = X[P == 1]

    optim_encoder = optim.Adam(model.encoder.parameters(), lr=lr)
    optim_decoder = optim.Adam(model.decoder.parameters(), lr=lr)
    print('Critic params:  ')
    for c in model.critic:
        print(c.parameters())
    optim_crit = optim.Adam(model.critic[0].parameters(), lr=0.1)

    l1_crit = nn.L1Loss(size_average=False)

    n_of_batch = int(len(X) / (batch_size * 2)) * n_iter

    for i in range(n_of_batch):
        X_n = X_0[np.random.choice(len(X_0), batch_size)]
        X_u = X_1[np.random.choice(len(X_1), batch_size)]
        if adv:
            w_dist_last = 0
            eps = 1
            #while w_dist <= 0:
            while eps >= 1e-3:
            #while True:
                for t in range(c_iter):
                    optim_crit.zero_grad()
                    w_dist = model.wdist(X_n, X_u)
                    loss = -w_dist
                    loss.backward(retain_graph=True)
                    optim_crit.step()
                    eps = np.abs(w_dist.data.item() - w_dist_last)

                    # keep training crit until distance no longer decrease
                    w_dist_last = w_dist.data.item()

                    for p in model.critic.parameters():
                        p.data.clamp_(-0.1, 0.1)


        # for t in range(c_iter):
        optim_encoder.zero_grad()
        optim_decoder.zero_grad()

        # only use the encoder g
        mse, wdist = model.forward(X_n, X_u)

        if adv:
            loss = mse + wdist * alpha
        else:
            loss = mse

        # L1 regularization
        reg_loss = 0
        #for param in model.encoder.parameters():
        #    reg_loss += torch.abs(param).sum()
        for layer in model.encoder:
            if type(layer) is nn.Linear:
                #norm = torch.sum(torch.pow(torch.sum(torch.abs(layer.weight), dim=0), 2))
                norm = 0.0
                for row in layer.weight.transpose(0,1):
                    norm += torch.sum(torch.pow(row, 2))
                reg_loss += norm

        loss += C_reg * reg_loss
        loss.backward(retain_graph=True)

        # use mse and wdist to update g and f
        optim_encoder.step()
        optim_decoder.step()

        text = 'mse: %.4f, critic: %.4f' % (mse.item(), wdist.item())
        if compute_emd:
            g_0 = model.encoder(X_u).detach().cpu().numpy()
            g_1 = model.encoder(X_n).detach().cpu().numpy()
            real_emd = emd_samples(g_0, g_1)
            text += ", emd: %.4f" % real_emd

        if verbose:
            update_progress(i, n_of_batch, time_s, text=text + ' ')


def cross_entropy(y, y_score):
    """
    Calculate the mean cross entropy.
        y: expected class labels.
        y_score: predicted class scores.
    Return: the cross entropy loss.
    """
    return -torch.mean(torch.mul(y, torch.log(y_score)) + torch.mul((1-y), torch.log(1-y_score)))


def train_cls(X, y, P, train_rate=0.7, c=0.0):
    """
    Train a classifier.
    The performance of the classifier is evaluated and printed.

    Parameters:
        X: input features.
        y: label.
        P: the protected attribute.
        train_rate: the ratio of the training data.
        c: the parameter specifying the inverse of regularization strength.
    """
    lin_model = nn.Sequential(nn.Linear(len(X[0]),1), nn.Sigmoid())
    lin_model.cuda()
    optimizer = optim.SGD(lin_model.parameters(), lr=0.01, weight_decay=c)
    train_len = int(train_rate * len(X))

    X = torch.tensor(X).cuda()
    y = torch.tensor(y).float().cuda()
    X_train = X[:train_len]
    y_train = y[:train_len]
    X_test = X[train_len+1:]
    y_test = y[train_len+1:]

    for i in range(1000):
        optimizer.zero_grad()
        y_score = lin_model(X_train)
        loss = cross_entropy(y_train, y_score)

        '''
        if c > 0:
            for p in lin_model.critic.parameters():
                p.data.clamp_(-c, c)
        '''

        loss.backward()
        optimizer.step()

    y_train_score = lin_model(X_train).cpu().data.numpy()
    y_test_score = lin_model(X_test).cpu().data.numpy()

    P = np.array(P)
    P_train = P[:train_len]
    P_test = P[train_len+1:]
    def get_score_ratio(scores, P_):
        scores_pos = sum(scores[P_==1]) / sum(P_==1)
        scores_neg = sum(scores[P_==0]) / sum(P_==0)
        print(scores_pos, scores_neg)
        return 1.0 * max(scores_pos, scores_neg) / min(scores_pos, scores_neg)

    print('train fair ratio: ' + str(get_score_ratio(y_train_score, P_train)))
    print('test fair ratio: ' + str(get_score_ratio(y_test_score, P_test)))
    print('train performance: ')
    print(evaluate_performance_sim(y_train.cpu().data.numpy(), y_train_score))
    print('test performance: ')
    print(evaluate_performance_sim(y_test.cpu().data.numpy(), y_test_score))
