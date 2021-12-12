import os
import time

# timestramp
time_start = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
timestramp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

## main ##
FILE_NAME    = 'ft06.txt'
OPT_MAKESPAN = 55
NUM_MACHINE  = 6
NUM_JOB      = 6
DIM_ACTION   = 12 #10
DIM_ACTION   = 10

RLS_RULE = 'SPT'

## arguments ##
DEVICE  = 'cuda'
MODEL   = f'model/djss_attention/ddqn-4x500x6-normalize-{timestramp}.pth'
# LOG_DIR = 'log/dqn'
LOG_DIR = 'log_djss_attention'
# WRITTER = f'log/DQN-3x6x6-ep99995-{time.time()}'
# WRITTER = f'log_djss/DDQN-20x100x6-{time_start}'
WRITTER = f'log_djss_attention/DDQN-4x500x6-{time_start}'
# train
EPISODE        = 50000
PRIORI_PERIOD  = 100 # 0# DIM_ACTION * 10
CAPACITY       = int(10000)
# CAPACITY       = int(5000)
WARMUP         = CAPACITY #/ 2
BATCH_SIZE     = 8#128
LEARNING_R     = .00005
GAMMA          = .99
EPS_DECAY      = 0.999999525 #.99982
# EPS_DECAY      = .999826#82
EPS_MIN        = .15 #.2 #.1
FREQ           = 16 #1 #4
# TARGET_FREQ	   = 1000 #300 #600 #500
TARGET_FREQ	   = 500 #500
RENDER_EPISODE = 900
# test
SEED         = 999 #2021111
SEED         = 20211211
TEST_EPSILON = .001
