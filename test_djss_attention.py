#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
##----- [Trainer] Main program of Learning ------
# * Author: Colin, Lee
# * Date: Aug 16th, 2021
##-----------------------------------------------
#
# from config import EPISODE
import os
import sys
import time
import torch
import random
import logging
import argparse
import itertools

import numpy    as np
import pandas   as pd
import torch.nn as nn
import matplotlib.pyplot    as plt

from tensorboardX import SummaryWriter
from datetime     import datetime as dt
from tqdm         import tqdm

# from simulation_env.env_jobshop_v1 import Factory
from simulation_env.env_for_job_shop_v7_attention import Factory
# from dqn_agent_djss                     import DQN as DDQN
from ddqn_agent_attention                         import DDQN

import pdb

# seed
import config_djss_attention as config
seed = config.SEED #999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.DEBUG)
plt.set_loglevel('WARNING') 


def test(args, _env, agent, writer, CHECK_PTS):
    logging.info('\n* Start Testing')
    env = _env

    action_space = env.action_space
    epsilon      = args.test_epsilon
    seeds        = [args.seed + i for i in range(100)]
    # seeds        = [args.seed + i for i in range(30)]
    rewards      = []
    makespans    = []
    lst_mean_ft  = []

    n_episode   = 0
    seed_loader = tqdm(seeds)
    ##################### Record Action ###################
    episode_percentage, episode_selection = [], []
    ######################################################

    for seed in seed_loader:
    #################### Record Action ###################
        action_selection = [0] * env.dim_actions
    ######################################################
        n_episode   += 1
        total_reward = 0
        # env.seed(seed)
        state = env.reset()
        for t in itertools.count(start=1):

            #action = agent.select_action(state, epsilon, action_space)
            action = agent.select_best_action(state)
            
            # execute action
            next_state, reward, done, _ = env.step(action)
    #################### Record Action ###################
            action_selection[action] += 1
    ######################################################

            state         = next_state
            total_reward += reward

            # env.render(done)
            #env.render(terminal=done)

            if done:
                writer.add_scalar(f'Test_{CHECK_PTS}/Episode_Reward'  , total_reward, n_episode)
                writer.add_scalar(f'Test_{CHECK_PTS}/Episode_Makespan', env.makespan, n_episode)
                writer.add_scalar(f'Test_{CHECK_PTS}/Episode_MeanFT', env.mean_flow_time, n_episode)
                rewards.append(total_reward)
                makespans.append(env.makespan)
                lst_mean_ft.append(env.mean_flow_time)

                # # Check the scheduling result
                # fig = env.gantt_plot.draw_gantt(env.makespan)
                # writer.add_figure('Test/Gantt_Chart', fig, n_episode)
                break

        env.close()
    #################### Record Action ###################
        # statistic of the selection of action 
        action_percentage = [0] * len(action_selection)
        for act in range(len(action_selection)):
            action_percentage[act] = action_selection[act] / t
        episode_selection.append(action_selection)
        episode_percentage.append(action_percentage)
    ######################################################
    df_act_res = pd.DataFrame(episode_selection)
    df_act_per = pd.DataFrame(episode_percentage)
    df_act_res.to_csv('testing_action_result.csv')
    df_act_per.to_csv('testing_action_percentage.csv')

    logging.info(f'  - Average Reward   = {np.mean(rewards)}')
    logging.info(f'  - Average Makespan = {np.mean(makespans)}')
    logging.info(f'  - Average MeanFT = {np.mean(lst_mean_ft)}')

    df = pd.DataFrame(lst_mean_ft)
    pd.set_option('display.max_rows', df.shape[0] + 1)
    print(df)
    df.to_csv('testing_result_09_4m200n.csv')


def main():
    import config_djss_attention as config
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default=config.DEVICE) #'cuda')
    parser.add_argument('-m', '--model' , default=config.MODEL) #'model/dqn.pth')
    parser.add_argument('--logdir'      , default=config.LOG_DIR) #'log/dqn')
    # train
    parser.add_argument('--warmup'        , default=config.WARMUP        , type=int)
    parser.add_argument('--episode'       , default=config.EPISODE       , type=int)
    parser.add_argument('--capacity'      , default=config.CAPACITY      , type=int)
    parser.add_argument('--batch_size'    , default=config.BATCH_SIZE    , type=int)
    parser.add_argument('--lr'            , default=config.LEARNING_R    , type=float)
    parser.add_argument('--eps_decay'     , default=config.EPS_DECAY     , type=float)
    parser.add_argument('--eps_min'       , default=config.EPS_MIN       , type=float)
    parser.add_argument('--gamma'         , default=config.GAMMA         , type=float)
    parser.add_argument('--freq'          , default=config.FREQ          , type=int)
    parser.add_argument('--target_freq'   , default=config.TARGET_FREQ   , type=int)
    parser.add_argument('--render_episode', default=config.RENDER_EPISODE, type=int)
    parser.add_argument('--priori_period' , default=config.PRIORI_PERIOD , type=int)
    # test
    parser.add_argument('--test_only'   , action='store_true')
    parser.add_argument('--render'      , action='store_true')
    parser.add_argument('--seed'        , default=config.SEED        , type=int)
    parser.add_argument('--test_epsilon', default=config.TEST_EPSILON, type=float)
    args = parser.parse_args()

    ## main ##
    file_name       = config.FILE_NAME
    file_dir        = os.getcwd() + '/simulation_env/instance'
    file_path       = os.path.join(file_dir, file_name)

    opt_makespan    = config.OPT_MAKESPAN
    num_machine     = config.NUM_MACHINE
    num_job         = config.NUM_JOB

    # rls_rule = config.RLS_RULE

    # Agent & Environment
    # env    = Factory(num_job, num_machine, file_path, opt_makespan, log=False)
    # env    = Factory(file_path, default_rule='FIFO', util=0.85, log=False)#True)
    env    = Factory(file_path, default_rule='FIFO', util=0.9, log=False)#True)
    # agent  = DQN(env.dim_observations, env.dim_actions, args)
    agent  = DDQN(env.dim_observations, env.dim_actions, args)

    # Tensorboard to trace the learning process
    ## -----------------------------------------------
    MODEL_VERSION = "20211224-145648"
    CHECK_PTS     = 19000
    # CHECK_PTS = 50000
    ## -----------------------------------------------
    # time_start = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
    # writer = SummaryWriter(f'log_djss_attention/DDQN-4x500x6-{time_start}')
    writter_name = config.WRITTER
    # writer       = SummaryWriter(f'{writter_name}')
    # writer = SummaryWriter(f'test_log/DDQN-attention-h4-load085-{MODEL_VERSION}')
    writer = SummaryWriter(f'test_log/DDQN-attention-h4-load09-{MODEL_VERSION}')
    # writer = SummaryWriter(f'log/DDQN-{time.time()}')
    ## Test ##  To test the pre-trained model
    # agent.load(args.model)
    # agent.load('model/djss_attention/ddqn-attention-h4-20211224-145648.pth-ck-19000.pth')
    agent.load(f'model/djss_attention/ddqn-attention-h4-{MODEL_VERSION}.pth-ck-{CHECK_PTS}.pth')
    # agent.load(f'model/djss_attention/ddqn-attention-h4-20211224-145648.pth')
    test(args, env, agent, writer, CHECK_PTS)

    writer.close()


if __name__ == '__main__':
    main()

