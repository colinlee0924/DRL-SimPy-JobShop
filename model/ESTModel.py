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
def __reset_param_impl__(cnn_net):
    """
    """
    # --- do init ---
    conv = cnn_net.conv
    n1   = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n1))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__() # necessary
        self.conv = nn.Conv2d(cin, cout, (3, 3), padding=1)
        self.bn   = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def reset_param(self):
        #normalize the para of cnn network
        __reset_param_impl__(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        super(Net, self).__init__()

        # convolution layers
        self.conv_1  = nn.Conv2d(3 , 32, kernel_size=3, stride=1)
        self.conv_2  = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn_32   = nn.BatchNorm2d(32)
        self.bn_64   = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU()
        self.flatten = nn.Flatten()

        #normalize the para of cnn network
        n1 = self.conv_1.kernel_size[0] * self.conv_1.kernel_size[1] * self.conv_1.out_channels
        n2 = self.conv_2.kernel_size[0] * self.conv_2.kernel_size[1] * self.conv_2.out_channels
        self.conv_1.weight.data.normal_(0, math.sqrt(2. / n1))
        self.conv_2.weight.data.normal_(0, math.sqrt(2. / n2))

        self.cnn = nn.Sequential(
            self.conv_1, self.bn_32, self.relu, #,
            self.conv_2, self.bn_64, self.relu
        )

        # check the output of cnn, which is [fc1_dims]
        self.cnn_outputs_len_1 = self.cnn_out_dim(state_dim[0])
        self.cnn_outputs_len_2 = self.cnn_out_dim(state_dim[1])

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

        cnn_out_1   = self.cnn(x[0])
        cnn_out_2   = self.cnn(x[1])
        cnn_out_1   = cnn_out_1.reshape(-1, self.cnn_outputs_len_1)
        cnn_out_2   = cnn_out_2.reshape(-1, self.cnn_outputs_len_2)
        fcn_input_1 = self.flatten(cnn_out_1)
        fcn_input_2 = self.flatten(cnn_out_2)
        fcn_input   = torch.cat((fcn_input_1, fcn_input_2), 0)
        actions     = self.fcn(fcn_input)
        return actions

    def cnn_out_dim(self, input_dims):
        return self.cnn(torch.zeros(1, *input_dims)
                       ).flatten().shape[0]

