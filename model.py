
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


class FairRep():
    def __init__(self, n_features, n_dim, n_protected=2, encoder=None, decoder=None, critic=None):
        self.n_prot = n_protected
        if encoder is None:
            self.encoder = nn.Sequential(nn.Linear(n_features, n_dim))
        else:
            self.encoder = encoder
        if decoder is None:
            self.decoder = nn.Sequential(nn.Linear(n_dim, n_features))
        else:
            self.decoder = decoder
        if critic is None:
            '''
            self.critic = nn.Sequential(nn.Linear(n_dim, n_dim),
                                    nn.ReLU(True),
                                    nn.Linear(n_dim,n_dim),
                                    nn.ReLU(True),
                                    nn.Linear(n_dim,n_dim),
                                    nn.ReLU(True),
                                    nn.Linear(n_dim,n_dim),
                                    nn.ReLU(True),
                                    nn.Linear(n_dim,1))
            '''
            #self.critic = [nn.Sequential(nn.Linear(n_dim, 5), nn.ReLU(), nn.Linear(5,1))
            #               for _ in range(n_protected)]
            self.critic = [nn.Sequential(nn.Linear(n_dim,1))] * n_protected
        else:
            self.critic = critic

    def wdist(self, x_0, x_1, p=0):
        c_0 = self.critic[p](self.encoder(x_0))
        c_1 = self.critic[p](self.encoder(x_1))
        w_dist = torch.mean(c_0 - c_1)
        return w_dist

    def cuda(self):
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()
        for t in range(self.n_prot):
            self.critic[t] = self.critic[t].cuda()

    def cpu(self):
        self.encoder.cpu()
        self.decoder.cpu()
        for t in range(self.n_prot):
            self.critic[t] = self.critic[t].cpu()

    def forward(self, x_0, x_rest, p=0):
        g_0 = self.encoder(x_0)
        g_1 = self.encoder(x_rest)

        # mse loss
        r_0 = self.decoder(g_0)
        r_1 = self.decoder(g_1)
        mse_0 = torch.mean(torch.pow(r_0 - x_0, 2))
        mse_1 = torch.mean(torch.pow(r_1 - x_rest, 2))
        mse = mse_0 + mse_1

        return mse, self.wdist(x_0, x_rest, p)
