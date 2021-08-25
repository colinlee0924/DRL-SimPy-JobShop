import os

## main ##
FILE_NAME    = 'job_info.xlsx'
OPT_MAKESPAN = 55
NUM_MACHINE  = 6
NUM_JOB      = 6
DIM_ACTION   = 12 #10

## arguments ##
DEVICE  = 'cuda'
MODEL   = 'model/dqn-ft06-aciton-12.pth'
LOG_DIR = 'log/dqn'
# train
WARMUP         = 10000
EPISODE        = 1200
CAPACITY       = 10000
BATCH_SIZE     = 128
LEARNING_R     = .0005
GAMMA          = .99
EPS_DECAY      = .9982
EPS_MIN        = .1
FREQ           = 4
TARGET_FREQ	   = 500
RENDER_EPISODE = 900
# test
SEED         = 2021111
TEST_EPSILON = .001