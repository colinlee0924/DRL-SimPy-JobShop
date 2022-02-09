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
from simulation_env.env_for_job_shop_v7_attention1 import Factory
# from dqn_agent_djss                     import DQN as DDQN
from ddqn_agent_attention_paper1                   import DDQN

import pdb
import wandb

# seed
import config_djss_attention_paper as config
seed = config.SEED #999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.DEBUG)
plt.set_loglevel('WARNING') 
# -----------------------------------------------

def train(args, _env, agent, writer):
    logging.info('* Start Training')

    env          = _env
    action_space = env.action_space

    total_step, epsilon, ewma_reward = 0, 1., 0.

    # Switch to train mode
    agent.train()

    #################### Record Action ###################
    episode_percentage, episode_selection = [], []
    ######################################################

    # Training until episode-condition
    for episode in range(args.episode):
        total_reward = 0
        done         = False
        state        = env.reset()
    #################### Record Action ###################
        action_selection = [0] * env.dim_actions
    ######################################################

        # While not terminate
        for t in itertools.count(start=1):
            # if args.render and episode > 700:
            #     env.render(done)
            #     time.sleep(0.0082)

            # select action
            if episode < args.priori_period:
                action = episode % env.dim_actions
                # action = action_space.sample()
            elif total_step < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)

            # execute action
            next_state, reward, done, _ = env.step(action)
            

    #################### Record Action ###################
            action_selection[action] += 1
   ######################################################

            # store transition
            agent.append(state, action, reward, next_state, done)
            if done or reward > 10:
                for _ in range(20):
                    agent.append(state, action, reward, next_state, done)

            # optimize the model
            loss = None
            if total_step >= args.warmup:
                loss = agent.update(total_step)
                if loss is not None:
                    writer.add_scalar('Train-Step/Loss', loss,
                                      total_step)

            # transit next_state --> current_state 
            state         = next_state
            total_reward += reward
            total_step  += 1

            if args.render and episode > args.render_episode:
                env.render(done)
                # time.sleep(0.0082)

            # Break & Record the performance at the end each episode
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train-Episode/Reward', total_reward,
                                  episode)
                writer.add_scalar('Train-Episode/Makespan', env.makespan,
                                  episode)
                writer.add_scalar('Train-Episode/MeanFT', env.mean_flow_time,
                                  episode)
                writer.add_scalar('Train-Episode/Epsilon', epsilon,
                                  episode)
                writer.add_scalar('Train-Step/Ewma_Reward', ewma_reward,
                                  total_step)
                wandb.log({
                    'Reward': total_reward, 
                    'Makespan': env.makespan,
                    'MeanFT': env.mean_flow_time,
                    'Epsilon': epsilon,
                    'Ewma_Reward': ewma_reward,
                    'Episode': episode
                    })

                if loss is not None:
                    writer.add_scalar('Train-Step/Loss', loss,
                                      total_step)
                    wandb.log({
                        'loss': loss
                        })
                logging.info(
                    '  - Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tMakespan: {:.2f}\tMeanFT: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_step, episode, t, total_reward, ewma_reward, env.makespan, env.mean_flow_time,
                            epsilon))

                # # Check the scheduling result
                # if episode % 50 == 0:
                #     fig = env.gantt_plot.draw_gantt(env.makespan)
                #     writer.add_figure('Train-Episode/Gantt_Chart', fig, episode)
                break
        
        # if episode > 1000:
            # epsilon = max(epsilon * args.eps_decay, args.eps_min)
        env.close()
        ## Train ##
        if episode % 1000 == 0 and episode > 0:
            agent.save(f'{args.model}-1ck-{episode}.pth')
    #################### Record Action ###################
        # statistic of the selection of action 
        action_percentage = [0] * len(action_selection)
        for act in range(len(action_selection)):
            action_percentage[act] = action_selection[act] / t
        episode_percentage.append(action_percentage)
        episode_selection.append(action_selection)
        action_list = [
               'FIFO'   , 'LIFO'   , 'SPT' , 'LPT', 
               'LWKR'   , 'MWKR'   , 'SSO' , 'LSO',
               'SPT+SSO', 'LPT+LSO' 
               ]
        for act in range(len(action_list)):
            wandb.log({
                action_list[act]: action_percentage[act],
                f'num_{action_list[act]}': action_selection[act]
                })
    ######################################################
    df_act_res = pd.DataFrame(episode_selection)
    df_act_per = pd.DataFrame(episode_percentage)
    df_act_res.to_csv('paper_training_action_result1.csv')
    df_act_per.to_csv('paper_training_action_percentage1.csv')


def test(args, _env, agent, writer):
    logging.info('\n* Start Testing')
    env = _env

    action_space = env.action_space
    epsilon      = args.test_epsilon
    seeds        = [args.seed + i for i in range(100)]
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
                writer.add_scalar('Test/Episode_Reward'  , total_reward, n_episode)
                writer.add_scalar('Test/Episode_Makespan', env.makespan, n_episode)
                writer.add_scalar('Test/Episode_MeanFT', env.mean_flow_time, n_episode)
                wandb.log({
                    'Test_Reward': total_reward, 
                    'Test_Makespan': env.makespan,
                    'Test_MeanFT': env.mean_flow_time
                    })

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
        episode_percentage.append(action_percentage)
        episode_selection.append(action_selection)
    ######################################################
    df_act_res = pd.DataFrame(episode_selection)
    df_act_per = pd.DataFrame(episode_percentage)
    df_act_res.to_csv('paper_testing_action_result1.csv')
    df_act_per.to_csv('paper_testing_action_percentage1.csv')

    logging.info(f'  - Average Reward   = {np.mean(rewards)}')
    logging.info(f'  - Average Makespan = {np.mean(makespans)}')
    logging.info(f'  - Average MeanFT = {np.mean(lst_mean_ft)}')


def main():
    import wandb
    import config_djss_attention_paper as config
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
    args.batch_size = 32

    wandb.init(project='DRL-SimPy-JSS', config=args)

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
    env    = Factory(file_path, default_rule='FIFO', util=0.9, log=False)#True)
    # agent  = DQN(env.dim_observations, env.dim_actions, args)
    agent  = DDQN(env.dim_observations, env.dim_actions, args)

    # Tensorboard to trace the learning process
    # time_start = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
    # writer = SummaryWriter(f'log_djss_attention/DDQN-4x500x6-{time_start}')
    writter_name = config.WRITTER
    writer       = SummaryWriter('1' + f'{writter_name}')
    # writer = SummaryWriter(f'log/DQN-{time.time()}')
    # writer = SummaryWriter(f'log/DDQN-{time.time()}')

    ## Train ##
    # agent.load('model/djss_attention/ddqn-attention-h4-20211224-125019.pth-ck-1000.pth')
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save('1'+args.model)

    ## Test ##  To test the pre-trained model
    agent.load(args.model)
    test(args, env, agent, writer)

    writer.close()


if __name__ == '__main__':
    main()
