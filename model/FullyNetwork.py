#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
##----- Model for Action-value Prediction ------
# * Author: Colin, Lee
# * Date: Aug 16th, 2021
##----------------------------------------------
import math
import torch
import random
import logging

import numpy    as np
import torch.nn as nn

from collections import deque
from datetime    import datetime as dt

import pdb

logging.basicConfig(level=logging.DEBUG)
# -----------------------------------------------


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        super(Net, self).__init__()

        self.relu    = nn.ReLU()
        self.flatten = nn.Flatten()

        # check the output of cnn, which is [fc1_dims]
        self.fcn_inputs_length = torch.zeros(1, *state_dim).flatten()

        # fully connected layers
        self.fc1 =  nn.Linear(self.fcn_inputs_length, hidden_dim)
        self.fc2 =  nn.Linear(hidden_dim            , action_dim)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

        self.fcn = nn.Sequential(
            self.fc1, self.relu,
            self.fc2
        )

    def forward(self, x):
        '''
            - x : tensor in shape of (N, state_dim)
        '''

        cnn_out   = x.reshape(-1, self.fcn_inputs_length)
        fcn_input = self.flatten(cnn_out)
        actions   = self.fcn(fcn_input)
        return actions

    def cnn_out_dim(self, input_dims):
        return self.cnn(torch.zeros(1, *input_dims)
                       ).flatten().shape[0]

