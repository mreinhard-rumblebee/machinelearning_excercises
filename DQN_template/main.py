from keras.models import Model
from keras.layers import Dense, Input
import tensorflow as tf
import gym

from collections import deque
import numpy as np
import random

import matplotlib.pyplot as plt
from numpy.core._simd import targets


class GymEnvironment:
    def __init__(self, env_id, monitor_dir, max_timesteps=100000):
        self.max_timesteps = max_timesteps

        self.env = gym.make(env_id)

    def trainDQN(self, agent, no_episodes):
        self.runDQN(agent, no_episodes, training=True)

        # Automatically save weights of trained network
        agent.model.save_weights("cartpole-v0.h5", overwrite=True)

    def runDQN(self, agent, no_episodes, training=False):
        count = 0
        rew = np.zeros(no_episodes)
        for episode in range(no_episodes):
            state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
            tot_rew = 0
            for t in range(self.max_timesteps):
                # TODO: Define the main DQN loop here. The observation of a transition is
                # already implemented. Use the functions defined in the agent's class.

                action = agent.select_action(state, training)

                # Execute the action and observe the transition which the environment gives you, i.e., next state
                # and reward
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])
                agent.record(state, action, reward, next_state, done)
                count += 1
                agent.update_weights(count)
                tot_rew += reward

                if done:
                    break
                # agent.update_weights(t)

            rew[episode] = tot_rew

            # TODO: Implement here a function that evaulates the agent's performance for every x episodes by
            # calling runDQN directly and returns an average of total rewards for 100 runs, if your objective is
            # reached, you can terminate training

            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, no_episodes, tot_rew, agent.epsilon))
        return rew


class DQN_Agent:
    def __init__(self, no_of_states, no_of_actions, load_old_model):
        self.state_size = no_of_states
        self.action_size = no_of_actions

        # Hyperparameters
        # TODO: Play around with these numbers once you have a running version of the code
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # eps-greedy exploration rate
        self.batch_size = 64  # maximum size of the batches sampled from memory
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.trainS = 300


        # TODO: Initialize the neural network models of weight and target weight networks
        self.model = self.nn_model((no_of_states,), no_of_actions, load_old_model)
        self.target_model = self.nn_model((no_of_states,), no_of_actions, load_old_model)

        # TODO: Define times at which target weights are synchronized
        self.target_model_time = 100

        # Maximal size of memory buffer
        self.memory = deque(maxlen=2000)

    def nn_model(self, state_size, action_size, load_old_model):
        # TODO: Define your neural network
        layer_input = Input(state_size)
        layer_one = Dense(512, input_shape=state_size, activation="relu", kernel_initializer='random_normal',
                          bias_initializer='zeros')(layer_input)
        layer_two = Dense(512, activation="relu", kernel_initializer='random_normal', bias_initializer='zeros')(layer_one)
        layer_three = Dense(512, activation="relu", kernel_initializer='random_normal', bias_initializer='zeros')(layer_two)
        layer_four = Dense(action_size, activation="linear", kernel_initializer='random_normal', bias_initializer='zeros')(layer_three)

        model = Model(inputs=layer_input, outputs=layer_four)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.00025), metrics=["accuracy"])
        return model

        # If you already have a set of weights from previous training, you can load them here
        if load_old_model == 1:
            model = self.model.load_weights("cartpole-v0.h5_target")
        return model

    def select_action(self, state, training=True):
        # TODO: Define the action selection. Don't forget to ensure exploration in case of training.
        if training:
            if np.random.random() <= self.epsilon:
                return random.randrange(self.action_size)
            else:
                return np.argmax(self.model.predict(state))
        else:
            return np.argmax(self.model.predict(state))

    # Here newly observed transitions are stored in the experience replay buffer
    # TODO maybe implement dynamic epsilon
    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.trainS:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # TODO Rework this function!
    def update_weights(self, t):  # t for time stepp at wich target moddle will be updated
        # TODO: Define the function to update weights of your network
        if len(self.memory) < self.batch_size:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, self.batch_size)

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.target_model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
                #target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))


        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
        if (t % self.target_model_time) == 0:
            self.target_model.set_weights(self.model.get_weights())


if __name__ == "__main__":
    environment = GymEnvironment('CartPole-v0', 'gymresults/cartpole-v0')

    # TODO: Define the number of states and actions - you will need this information for defining your neural network
    no_of_states = 4
    no_of_actions = 2

    # If you want to load weights of an already trained model, set this flag to 1
    load_old_model = 0

    # The agent is initialized
    agent = DQN_Agent(no_of_states, no_of_actions, load_old_model)

    # Train your agent
    no_episodes = 400
    environment.trainDQN(agent, no_episodes)

    # Run your agent
    no_episodes_run = 100
    rew = environment.runDQN(agent, no_episodes_run)

    # TODO: Implement here a function visualizing/plotting, e.g.,
    plotting = False
    if plotting:

    # your agent's performance over the number of training episodes
        iterations = range(0, no_episodes + 1, no_episodes_run)
        plt.plot(iterations, rew)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.ylim(top=250)
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
