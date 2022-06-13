
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import gym

from collections import deque
import numpy as np
import random

import matplotlib.pyplot as plt


class GymEnvironment:
    def __init__(self, env_id, monitor_dir, max_timesteps=100000):
        self.max_timesteps = max_timesteps
        self.env = gym.make(env_id)

    def trainDQN(self, agent, no_episodes):
        self.runDQN(agent, no_episodes, training=True)

        # Automatically save weights of trained network
        agent.model.save_weights("cartpole-v0.h5", overwrite=True)


    def runDQN(self, agent, no_episodes, training=False):

        rew = np.zeros(no_episodes)
        for episode in range(no_episodes):
            state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
            tot_rew = 0
            for t in range(self.max_timesteps):
                #TODO: Define the main DQN loop here. The observation of a transition is
                # already implemented. Use the functions defined in the agent's class.

                action = agent.select_action(state, training)

                # Execute the action and observe the transition which the environment gives you, i.e., next state
                # and reward
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])

                tot_rew += reward

                if done:
                    break

            rew[episode] = tot_rew

            # TODO: Implement here a function that evaulates the agent's performance for every x episodes by
            # calling runDQN directly and returns an average of total rewards for 100 runs, if your objective is
            # reached, you can terminate training

            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, no_episodes, tot_rew, agent.epsilon))
        return rew


class DQN_Agent:
    def __init__(self, no_of_states, no_of_actions,load_old_model):
        self.state_size = no_of_states
        self.action_size = no_of_actions

        # Hyperparameters
        #TODO: Play around with these numbers once you have a running version of the code
        self.gamma =   # discount rate
        self.epsilon =   # eps-greedy exploration rate
        self.batch_size =   # maximum size of the batches sampled from memory

        #TODO: Initialize the neural network models of weight and target weight networks
        self.model =
        self.target_model =

        # TODO: Define times at which target weights are synchronized
        self.target_model_time =

        # Maximal size of memory buffer
        self.memory = deque(maxlen=2000)

    def nn_model(self,state_size,action_size,load_old_model):
        # TODO: Define your neural network
        model =


        # If you already have a set of weights from previous training, you can load them here
        if load_old_model == 1:
            model = self.model.load_weights("cartpole-v0.h5_target")
        return model

    def select_action(self, state, training=True):
        # TODO: Define the action selection. Don't forget to ensure exploration in case of training.
        act =
        return act

    # Here newly observed transitions are stored in the experience replay buffer
    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_weights(self,t):
    # TODO: Define the function to update weights of your network


if __name__ == "__main__":
    environment = GymEnvironment('CartPole-v0', 'gymresults/cartpole-v0')


    #TODO: Define the number of states and actions - you will need this information for defining your neural network
    no_of_states =
    no_of_actions =

    # If you want to load weights of an already trained model, set this flag to 1
    load_old_model = 0

    # The agent is initialized
    agent = DQN_Agent(no_of_states,no_of_actions,load_old_model)

    # Train your agent
    no_episodes =
    environment.trainDQN(agent, no_episodes)

    # Run your agent
    no_episodes_run =
    rew = environment.runDQN(agent, no_episodes_run)

    # TODO: Implement here a function visualizing/plotting, e.g.,
    # your agent's performance over the number of training episodes

    # Here you can watch a simulation on how your agent performs after being trained.
    # NOTE that this part will try to load an existing set of weights, therefore set visualize_agent to TRUE, when you
    # already saved a set of weights from a training session
    visualize_agent = False
    if visualize_agent == True:
        env = gym.make('CartPole-v0')
        load_model = 1
        state_size = 4
        action_size = 2
        agent = DQN_Agent(state_size, action_size, load_model)
        state = env.reset().reshape(1, env.observation_space.shape[0])
        for _ in range(1000):
            env.render()
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, env.observation_space.shape[0])
            state = next_state
        env.close()
