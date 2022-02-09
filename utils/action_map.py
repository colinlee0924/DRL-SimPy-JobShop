# file name : action_map.py
import sys
if '..' not in sys.path:
    sys.path.append('..')

import config

ACTION_MAP  = {}
dim_actions = config.DIM_ACTION

for action in range(dim_actions):
    if action == 0:
        dspch_rule = 'FIFO'
    elif action == 1:
        dspch_rule = 'LIFO'
    elif action == 2:
        dspch_rule = 'SPT'
    elif action == 3:
        dspch_rule = 'LPT'
    elif action == 4:
        dspch_rule = 'LWKR'
    elif action == 5:
        dspch_rule = 'MWKR'
    elif action == 6:
        dspch_rule = 'SSO'
    elif action == 7:
        dspch_rule = 'LSO'
    elif action == 8:
        dspch_rule = 'SPT+SSO'
    elif action == 9:
        dspch_rule = 'LPT+LSO'
    elif action == 10:
        dspch_rule = 'STPT'
    elif action == 11:
        dspch_rule = 'LTPT'
    
    ACTION_MAP[action] = dspch_rule