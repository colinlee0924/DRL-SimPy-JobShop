#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
##----- [Algorithm] DQN Implementation ------
# * Author: Colin, Lee
# * Date: Aug 16th, 2021
##-------------------------------------------
#
import time
import torch
import random
import logging
import argparse
import itertools

import numpy    as np
import torch.nn as nn

from tensorboardX import SummaryWriter
from datetime     import datetime as dt

from model.NetworkModel_attention_paper1 import MultiHeadRelationalModule as Net
# from model.FullyNetwork import Net
from utils.MemeryBuffer import ReplayMemory

import pdb
import config_djss_attention as config
seed = config.SEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.DEBUG)
# -----------------------------------------------

class DDQN:
    def __init__(self, dim_state, dim_action, args):
        ## config ##
        self.device      = torch.device(args.device)
        self.batch_size  = args.batch_size
        self.gamma       = args.gamma
        self.freq        = args.freq
        self.target_freq = args.target_freq

        self._behavior_net = Net(dim_state, dim_action, args.device).to(self.device)
        self._target_net   = Net(dim_state, dim_action, args.device).to(self.device)
        # -------------------------------------------
        # initialize target network
        # -------------------------------------------
        self._target_net.load_state_dict(self._behavior_net.state_dict())#, map_location=self.device)

        # self._optimizer = torch.optim.RMSprop(
        self._optimizer = torch.optim.Adam(
                            self._behavior_net.parameters(), 
                            lr=args.lr
                          )
        self._criteria  = nn.MSELoss()
        # self._criteria  = nn.SmoothL1Loss()
        # memory
        self._memory    = ReplayMemory(capacity=args.capacity)

    def select_best_action(self, state):
        '''
            - state: (state_dim, )
        '''
        during_train = self._behavior_net.training
        if during_train:
            self.eval()
        state  = torch.Tensor(state).to(self.device)
        state  = DDQN.reshape_input_state(state)
        with torch.no_grad():
            qvars  = self._behavior_net(state)      # (1, act_dim)
            action = torch.argmax(qvars, dim=-1)    # (1, )

        if during_train:
            self.train()

        return action.item()

    def select_action(self, state, epsilon, action_space):
        '''
        epsilon-greedy based on behavior network

            -state = (state_dim, )
        '''
        if random.random() < epsilon:
            return action_space.sample()
        else:
            return self.select_best_action(state)

    def append(self, state, action, reward, next_state, done):
        self._memory.append(
            state, 
            [action], 
            [reward],# / 10], 
            next_state,
            [1 - int(done)]
        )

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            return self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            return self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        ret = self._memory.sample(self.batch_size, self.device)
        state, action, reward, next_state, tocont = ret 

        q_values = self._behavior_net(state)                                    # (N, act_dim)
        q_value  = torch.gather(input=q_values, dim=-1, index=action.long())    # (N, 1)
        with torch.no_grad():
            ## Where DDQN is different from DQN
            qs_next_behavior_net  = self._behavior_net(next_state)                              # (N, act_dim)
            indx_behavior_actions = torch.argmax(qs_next_behavior_net, dim=1).unsqueeze(dim=1)
            qs_next               = self._target_net(next_state)                                # (N, act_dim)

            # compute V*(next_states) using predicted next q-values
            q_next                = qs_next.gather(dim=1, index=indx_behavior_actions)     # (N, 1)
            q_target              = gamma*tocont*q_next.detach() + reward.detach()

        loss = self._criteria(q_value, q_target.detach())

        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 100)#5)#10)
        self._optimizer.step()

        return loss.item()

    def _update_target_network(self):
        '''
        update target network by copying from behavior network
        '''
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        return None

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path, map_location=self.device)
        self._behavior_net.load_state_dict(model['behavior_net'])#, map_location=self.device)
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])#, map_location=self.device)
            self._optimizer.load_state_dict(model['optimizer'])#  , map_location=self.device)

    def train(self):
        self._behavior_net.train()
        self._target_net.eval()

    def eval(self):
        self._behavior_net.eval()
        self._target_net.eval()
    
    @staticmethod
    def reshape_input_state(state):
        state_shape = len(state.shape)
        state = state.unsqueeze(0)

        return state
