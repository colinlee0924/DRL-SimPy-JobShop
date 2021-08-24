#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
##----- [Utils] Memory Buffer ------
# * Author: Colin, Lee
# * Date: Aug 16th, 2021
##----------------------------------
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

class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, not_done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))