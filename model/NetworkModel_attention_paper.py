#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
##----- Model for Action-value Prediction ------
# * Author: Colin, Lee
# * Date: Aug 16th, 2021
##----------------------------------------------
import sys
if '..' not in sys.path:
    sys.path.append('..')
    
import math
import torch
import random
import logging

import numpy    as np
import torch.nn as nn


from collections import deque
from einops      import rearrange
from datetime    import datetime as dt

import pdb
import config_djss_attention as config

seed = config.SEED #999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

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


class MultiHeadRelationalModule(torch.nn.Module):
    def __init__(self, input_dim=(4, 100, 6), output_dim=10 ,device='cuda'):
        super(MultiHeadRelationalModule, self).__init__()
        self.device   = torch.device(device)
        self.conv1_ch = 16#64
        self.conv2_ch = 32
        # self.conv3_ch = 24
        # self.conv4_ch = 30
        self.ch_in        = input_dim[0]
        self.n_heads      = 4
        self.node_size    = 64 # dimension of nodes after passing the relational module
        self.sp_coord_dim = 2
        self.input_height = input_dim[1]
        self.input_width  = input_dim[2]
        self.n_cout_pixel = 596#int(self.input_height * self.input_width) #number of nodes (pixels num after passing the cnn)
        self.out_dim      = output_dim
        # self.lin_hid = 100
        # self.conv2_ch = self.ch_in
        self.conv2_ch = 32

        self.proj_shape = (
            (self.conv2_ch + self.sp_coord_dim), 
            (self.n_heads * self.node_size)
        )
        self.node_shape = (
            self.n_heads, 
            self.n_cout_pixel, 
            self.node_size
        )

        self.conv_1 = nn.Conv2d(self.ch_in   , self.conv1_ch, kernel_size=(2,2), padding=0) #A
        self.conv_2 = nn.Conv2d(self.conv1_ch, self.conv2_ch, kernel_size=(2,2), padding=0)
        #normalize the para of cnn network
        n1 = self.conv_1.kernel_size[0] * self.conv_1.kernel_size[1] * self.conv_1.out_channels
        n2 = self.conv_2.kernel_size[0] * self.conv_2.kernel_size[1] * self.conv_2.out_channels
        self.conv_1.weight.data.normal_(0, math.sqrt(2. / n1))
        self.conv_2.weight.data.normal_(0, math.sqrt(2. / n2))

        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.k_lin = nn.Linear(self.node_size   , self.n_cout_pixel) #B
        self.q_lin = nn.Linear(self.node_size   , self.n_cout_pixel)
        self.a_lin = nn.Linear(self.n_cout_pixel, self.n_cout_pixel)

        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        
        self.linear1 = nn.Linear((self.n_heads * self.node_size), self.node_size)
        self.norm1   = nn.LayerNorm([self.n_cout_pixel, self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

        self.relu    = nn.ReLU()

        self.cnn = nn.Sequential(
            self.conv_1, self.relu,
            self.conv_2, self.relu
        )

    def forward(self, x):
        N, Cin, H, W = x.shape
        x = self.cnn(x) 
        # # for visualization
        # with torch.no_grad(): 
        #     self.conv_map = x.clone() #C

        # Appends the (x, y) coordinates of each node to its feature vector and normalized to [0,1]
        _, _, cH, cW   = x.shape
        xcoords        = torch.arange(cW).repeat(cH, 1).float() / cW
        ycoords        = torch.arange(cH).repeat(cW, 1).transpose(1, 0).float() / cH
        spatial_coords = torch.stack([xcoords, ycoords], dim=0)
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N, 1, 1, 1)

        x = torch.cat([x, spatial_coords.to(self.device)], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.flatten(1, 2)
        
        # K, Q, V
        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K) 
        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q) 
        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V) 

        del x
        
        # Compatibility function
        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K)) #D
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A, dim=3) 
        # # for visualization
        # with torch.no_grad():
        #     self.att_map = A.clone() #E

        del Q; del K

        # Multi-head attention
        E = torch.einsum('bhfc, bhcd->bhfd', A, V) #F
        E = rearrange(E, 'b head n d -> b n (head d)')

        del A; del V

        # Linear forward
        E = self.linear1(E)
        E = self.relu(E)
        E = self.norm1(E)
        E = E.max(dim=1)[0]
        y = self.linear2(E)
        return torch.nn.functional.elu(y)
        # y = torch.nn.functional.elu(y)
        # return y


class Net(nn.Module):
    def __init__(self, state_dim=(3, 8, 8), action_dim=4, hidden_dim=128):
        super(Net, self).__init__()
        cin = state_dim[0]
        # convolution layers
        self.conv_1  = nn.Conv2d(cin , 64, kernel_size=(1,3), padding=1, stride=1)
        self.conv_2  = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.conv_3  = nn.Conv2d(32, 16, kernel_size=2, stride=1)
        self.bn_16   = nn.BatchNorm2d(16)
        self.bn_32   = nn.BatchNorm2d(32)
        self.bn_64   = nn.BatchNorm2d(64)
        self.relu    = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        #normalize the para of cnn network
        n1 = self.conv_1.kernel_size[0] * self.conv_1.kernel_size[1] * self.conv_1.out_channels
        n2 = self.conv_2.kernel_size[0] * self.conv_2.kernel_size[1] * self.conv_2.out_channels
        n3 = self.conv_3.kernel_size[0] * self.conv_3.kernel_size[1] * self.conv_3.out_channels
        self.conv_1.weight.data.normal_(0, math.sqrt(2. / n1))
        self.conv_2.weight.data.normal_(0, math.sqrt(2. / n2))
        self.conv_3.weight.data.normal_(0, math.sqrt(2. / n3))

        self.cnn = nn.Sequential(
            self.conv_1, self.bn_64, self.relu,
            self.conv_2, self.bn_32, self.relu,
            self.conv_3, self.bn_16, self.relu
        )

        # check the output of cnn, which is [fc1_dims]
        self.fcn_inputs_length = self.cnn_out_dim(state_dim)

        # fully connected layers
        self.fc1 =  nn.Linear(self.fcn_inputs_length, hidden_dim)
        self.fc2 =  nn.Linear(hidden_dim            , action_dim)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        # self.bn_fc2 = nn.BatchNorm1d(64)

        self.fcn = nn.Sequential(
            self.fc1, self.bn_fc1, self.relu,
            self.fc2
        )

    def forward(self, x):
        '''
            - x : tensor in shape of (N, state_dim)
        '''

        cnn_out   = self.cnn(x)
        cnn_out   = cnn_out.reshape(-1, self.fcn_inputs_length)
        fcn_input = self.flatten(cnn_out)
        actions   = self.fcn(fcn_input)
        return actions

    def cnn_out_dim(self, input_dims):
        return self.cnn(torch.zeros(1, *input_dims)
                       ).flatten().shape[0]

