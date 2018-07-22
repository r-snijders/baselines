#!/usr/bin/env python

import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

from matplotlib import pyplot as plt
from threading import Thread, Lock
import logging
import time
import argparse
    


class DQN:

    def __init__(self, args):
        self._args = args
        self._reward_buffer_mutex = Lock()
        self._reward_buffer = []
        self._reward_buffer_changed = True
        self._clicked_y = None
        self._clicked_x = None
        self._render_reward_threshold = 1000

    def model(self, inpt, num_actions, scope, reuse=False):
        """This model takes as input an observation and returns values of all actions."""
        with tf.variable_scope(scope, reuse=reuse):
            out = inpt
            out = layers.fully_connected(out, num_outputs=self._args.hidden_layer_size, activation_fn=tf.nn.tanh)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            return out

    def learn(self):

        with U.make_session(8):
            # Create the environment
            env = gym.make(self._args.env)
            # Create all the functions necessary to train the model
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
                q_func=self.model,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=self._args.learning_rate),
            )
            # Create the replay buffer
            replay_buffer = ReplayBuffer(self._args.replay_buffer_size)
            # Create the schedule for exploration starting from 1 till min_exploration_rate.
            exploration = LinearSchedule(schedule_timesteps=self._args.exploration_duration, initial_p=1.0, final_p=self._args.min_exploration_rate)

            # Initialize the parameters and copy them to the target network.
            U.initialize()
            update_target()

            episode_rewards = [0.0]
            obs = env.reset()
            for t in itertools.count():
                # Take action and update exploration to the newest value
                action = act(obs[None], update_eps=exploration.value(t))[0]
                new_obs, rew, done, _ = env.step(action)
                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    obs = env.reset()
                    episode_rewards.append(0)

                mean_episode_reward = np.mean(episode_rewards[-101:-1])
                # Show learned agent:
                if mean_episode_reward >= self._render_reward_threshold:
                    env.render() 

                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

                if done and len(episode_rewards) % 10 == 0:
                    self._reward_buffer_mutex.acquire()
                    self._reward_buffer.append(mean_episode_reward)

                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", len(episode_rewards))
                    logger.record_tabular("mean episode reward", round(mean_episode_reward, 1))
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    logger.dump_tabular()

                    self._reward_buffer_changed = True
                    self._reward_buffer_mutex.release()

    def onclick(self, event):
        self._clicked_y = event.ydata
        if self._clicked_y is not None:
            self._render_reward_threshold = self._clicked_y
            print("Setting visualization limit to: mean episode reward > %.2f" % self._render_reward_threshold)

    def plot(self, updated=True):
        plt.clf()

        plt.xlabel("Episodes")
        plt.ylabel("Mean Reward \n (over the last 100 episodes)")

        min_y, max_y = (-200, 200)
        if self._args.env == "CartPole-v0":
            min_y = 0.0
        if len(self._reward_buffer) > 0:
            data_min_y, data_max_y = min(self._reward_buffer), max(self._reward_buffer)
            if data_min_y < min_y:
                min_y = data_min_y 
            if data_max_y > max_y:
                max_y = data_max_y 
        plt.ylim((min_y, max_y))
        plt.plot([(r + 1) * 10 for r in list(range(len(self._reward_buffer)))], self._reward_buffer)

        if self._clicked_y is not None:
            plt.axhline(self._clicked_y, color='r')

        plt.gcf().autofmt_xdate()
        plt.tight_layout()


    def run(self):
        Thread(target=self.learn).start()
        #Draw and block:
        self.plt_drawer()

    def plt_drawer(self):
        plt.ion()
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        while True:
            self._reward_buffer_mutex.acquire()

            self.plot(self._reward_buffer_changed)
            self._reward_buffer_changed = False

            plt.draw()
            plt.pause(0.001)
            self._reward_buffer_mutex.release()
            time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(description="Modified version of custom_cartpol.py with visualization of mean episode rewards.")
    parser.add_argument('--env', help='Gym environment ID (examples: "CartPole-v0", "LunarLander-v2", "Acrobot-v1", "MountainCar-v0")', default='CartPole-v0')
    parser.add_argument('-er', dest="min_exploration_rate", type=float, default=0.02, help="Minimal exploration rate after exploration phase (default=0.02).")
    parser.add_argument('-ed', dest="exploration_duration", type=int, default=10000, help="Exploration phase duration in timesteps (default=10000).")
    parser.add_argument('-lr', dest="learning_rate", type=float, default=0.0005, help="Minimal exploration rate after exploration phase (default=0.0005).")
    parser.add_argument('-rs', dest="replay_buffer_size", type=int, default=5000, help="Size of replay buffer (default=5000).")
    parser.add_argument('-hl', dest="hidden_layer_size", type=int, default=64, help="Size of hidden layer of the MLP being used (default=64).")
    args = parser.parse_args()

    DQN(args).run()

if __name__ == "__main__":
    main()



