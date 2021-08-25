# file name : action_map.py

dim_actions = 10
ACTION_MAP  = {}

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
    
    ACTION_MAP[action] = dspch_rule