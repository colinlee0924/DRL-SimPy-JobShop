#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
##----- [Trainer] Main program of Learning ------
# * Author: Colin, Lee
# * Date: Aug 16th, 2021
##-----------------------------------------------
#
import os
import sys
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

from simulation_env.env_jobshop_v0 import Factory
from dqn_agent                     import DQN

import pdb

logging.basicConfig(level=logging.DEBUG)
# -----------------------------------------------

def train(args, _env, agent, writer):
    logging.info('* Start Training')

    env          = _env
    action_space = env.action_space

    total_steps, epsilon, ewma_reward = 0, 1., 0.

    # Switch to train mode
    agent.train()

    # Training until episode-condition
    for episode in range(args.episode):
        total_reward = 0
        state        = env.reset()

        # While not terminate
        for t in itertools.count(start=1):
            if args.render and episode > 700:
                env.render()
                time.sleep(0.0082)

            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)

            # execute action
            next_state, reward, done, _ = env.step(action)

            # store transition
            agent.append(state, action, reward, next_state, done)

            # optimize the model
            loss = None
            if total_steps >= args.warmup:
                loss = agent.update(total_steps)

            # transit next_state --> current_state 
            state         = next_state
            total_reward += reward
            total_steps  += 1

            env.render(done)

            # Break & Record the performance at the end each episode
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode_Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma_Reward', ewma_reward,
                                  total_steps)
                writer.add_scalar('Train/Episode_Makespan', env.makespan,
                                  total_steps)
                if loss is not None:
                    writer.add_scalar('Train/Loss', loss,
                                    total_steps)
                logging.info(
                    '  - Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tMakespan: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward, env.makespan,
                            epsilon))
                break
    env.close()


def test(args, _env, agent, writer):
    logging.info('* Start Testing')
    env = _env

    action_space = env.action_space
    epsilon      = args.test_epsilon
    seeds        = (args.seed + i for i in range(10))
    rewards      = []
    makespans    = []

    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        # env.seed(seed)
        state = env.reset()
        done = False
        for t in itertools.count(start=1):
            env.render(done)
            time.sleep(0.03)

            #action = agent.select_action(state, epsilon, action_space)
            action = agent.select_best_action(state)
            
            # execute action
            next_state, reward, done, _ = env.step(action)

            state         = next_state
            total_reward += reward

            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                rewards.append(total_reward)
                makespans.append(env.makespan)
                break
    logging.info('  - Average Reward   =', np.mean(rewards))
    logging.info('  - Average Makespan =', np.mean(makespans))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model' , default='model/dqn.pth')
    parser.add_argument('--logdir'      , default='log/dqn')
    # train
    parser.add_argument('--warmup'     , default=10000, type=int)
    parser.add_argument('--episode'    , default=1200 , type=int)
    parser.add_argument('--capacity'   , default=10000, type=int)
    parser.add_argument('--batch_size' , default=128  , type=int)
    parser.add_argument('--lr'         , default=.0005, type=float)
    parser.add_argument('--eps_decay'  , default=.9982, type=float)
    parser.add_argument('--eps_min'    , default=.1   , type=float)
    parser.add_argument('--gamma'      , default=.99  , type=float)
    parser.add_argument('--freq'       , default=4    , type=int)
    parser.add_argument('--target_freq', default=500  , type=int)
    # test
    parser.add_argument('--test_only'   , action='store_true')
    parser.add_argument('--render'      , action='store_true')
    parser.add_argument('--seed'        , default=2021111, type=int)
    parser.add_argument('--test_epsilon', default=.001   , type=float)
    args = parser.parse_args()

    ## main ##
    usr_interaction = False
    opt_makespan    = 55
    file_name       = 'job_info.xlsx'
    file_dir        = os.getcwd() + '/simulation_env/input_data'
    file_path       = os.path.join(file_dir, file_name)

    # Agent & Environment
    env    = Factory(6, 6, file_path, opt_makespan, log=False)
    agent  = DQN(env.dim_observations, env.dim_actions, args)

    # Tensorboard to trace the learning process
    writer = SummaryWriter(f'log/DQN-{time.time()}')

    ## Train ##
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)

    ## Test ##  To test the pre-trained model
    agent.load(args.model)
    test(args, env, agent, writer)

    writer.close()


if __name__ == '__main__':
    main()
