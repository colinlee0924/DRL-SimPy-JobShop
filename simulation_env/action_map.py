from itertools import permutations as pm

NUM_RULE   = 2
NUM_MACINE = 6
ACTION_MAP = {
               1: ['FIFO', 'FIFO', 'FIFO', 'FIFO', 'FIFO', 'FIFO'],
               2: ['FIFO', 'FIFO', 'FIFO', 'FIFO', 'FIFO', 'SPT'],
               3: ['FIFO', 'FIFO', 'FIFO', 'FIFO', 'SPT' , 'FIFO'],
               4: ['FIFO', 'FIFO', 'FIFO', 'SPT' , 'FIFO', 'FIFO'],
               5: ['FIFO', 'FIFO', 'SPT' , 'FIFO', 'FIFO', 'FIFO'],
               6: ['FIFO', 'SPT' , 'FIFO', 'FIFO', 'FIFO', 'FIFO'],
               7: ['SPT' , 'FIFO', 'FIFO', 'FIFO', 'FIFO', 'FIFO'],
              }

# def permute(lst_elements):
#     lst_permuted = []
#     for res in pm(lst_elements, len(lst_elements)):
#         lst_permuted.append(list(res))

#     return lst_permuted

# ACTION_MAP = {}

# for action in range(NUM_RULE ** NUM_MACINE):
#     ACTION_MAP[action] = ['FIFO', 'FIFO', 'FIFO', 'FIFO', 'FIFO', 'FIFO']
#     machine = 0
#     if action > 0 and action <= 6:
#         ACTION_MAP[action][machine] = 'FIFO'
