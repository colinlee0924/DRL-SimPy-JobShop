import os

## main ##
FILE_NAME    = 'job_info.xlsx'
OPT_MAKESPAN = 55
NUM_MACHINE  = 6
NUM_JOB      = 6

## arguments ##
DEVICE  = 'cuda'
MODEL   = 'model/dqn.pth'
LOG_DIR = 'log/dqn'
# train
WARMUP  = 10000
EPISODE = 1200