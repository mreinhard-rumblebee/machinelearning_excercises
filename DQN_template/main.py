from keras.models import Model
from keras.layers import Dense, Input
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
        self.env._max_episode_steps = 300

    def trainDQN(self, agent, no_episodes, visualize_agent=False):
        rew = self.runDQN(agent, no_episodes, training=True, visualize_agent=visualize_agent)

        # Automatically save weights of trained network
        agent.model.save_weights("cartpole-v0.h5", overwrite=True)
        return rew

    def runDQN(self, agent, no_episodes, training=False, visualize_agent=False, evaluation=False):
        rew = np.zeros(no_episodes)
        for episode in range(no_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, agent.state_size])
            done = False
            i = 0
            rwd = 0
            while not done:
                #possibility to visualize training or testing
                if visualize_agent:
                    self.env.render()
                action = agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, agent.state_size])
                rwd += reward
                # introduce negative rewards to incentivice agent to finish
                if not done or i == self.max_timesteps - 1:
                    reward = reward
                else:
                    reward = -100
                # in case of training record the observed tuple and adjust the epsilon rate that determines greedy behavior
                if training:
                    agent.record(state, action, reward, next_state, done)
                    agent.update_epsilon()
                # Set next state as current state to prepare for next state action pair
                state = next_state
                i += 1
                # update weights in case of training every 10 time steps
                if training:
                    agent.update_weights(i)
                if done:
                    break
            rew[episode] = rwd

            # TODO: Implement here a function that evaulates the agent's performance for every x episodes by
            # calling runDQN directly and returns an average of total rewards for 100 runs, if your objective is
            # reached, you can terminate training
            if episode % 50 == 0 and training and episode != 0:
                print("-------------------------------------------")
                print("Evaluation starts:")
                rew_subset = self.runDQN(agent, 100, training=False, evaluation=True)
                avg_subset = sum(rew_subset) / 100
                print("Average over 100 episodes: {}".format(
                    avg_subset))
                if avg_subset >= 200:
                    break
            # if evaluating, just plot the progress
            if not evaluation:
                if training:
                    print("episode: {}/{} | score: {} | e: {:.3f}".format(
                        episode + 1, no_episodes, rwd, agent.epsilon))
                else:
                    print("episode: {}/{} | score: {}".format(
                        episode + 1, no_episodes, rwd))
            else:
                if episode % 10 == 0:
                    print("Progress: {} %".format(episode))
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
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99

        # TODO: Initialize the neural network models of weight and target weight networks
        self.model = self.nn_model((no_of_states,), no_of_actions, load_old_model)
        self.target_model = self.nn_model((no_of_states,), no_of_actions, load_old_model)

        # TODO: Define times at which target weights are synchronized
        self.target_model_time = 12

        # Maximal size of memory buffer
        self.memory = deque(maxlen=2000)

    def nn_model(self, state_size, action_size, load_old_model):
        # TODO: Define your neural network
        layer_input = Input(state_size)
        layer_one = Dense(512, input_shape=state_size, activation="relu")(layer_input)
        layer_two = Dense(250, activation="relu")(layer_one)
        layer_three = Dense(30, activation="relu")(layer_two)
        layer_four = Dense(action_size, activation="linear")(layer_three)

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
            random_nr = np.random.random()
            if random_nr <= self.epsilon:
                act = random.randrange(self.action_size)
            else:
                act = np.argmax(self.model.predict(state))
        else:
            act = np.argmax(self.model.predict(state))
        return act

    # Here newly observed transitions are stored in the experience replay buffer
    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # update epsilon as long as its larger than our minvalue
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_weights(self, t):  # t for time step at wich target model will be updated
        # TODO: Define the function to update weights of your network
        #check whether there is enough samples for the minibatch
        if len(self.memory) < self.batch_size:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, self.batch_size)
        #Initialze Lists for Minibatch storage for each element stored in a sample
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        index = 0
        while index < self.batch_size:
            sample = minibatch[index]
            state[index] = sample[0]
            action.append(sample[1])
            reward.append(sample[2])
            next_state[index] = sample[3]
            done.append(sample[4])
            index += 1

        # predict the whole batch
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_next_target_model = self.target_model.predict(next_state)

        index = 0
        while index < self.batch_size:
            if not done[index]:
                target[index][action[index]] = reward[index] + self.gamma * (target_next_target_model[index][np.argmax(target_next[index])])
            else:
                target[index][action[index]] = reward[index]
            index += 1

        # fit model
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

        #update target model each target_model_time time steps
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
    no_episodes = 100
    visualize_agent_train = True
    rew_train = environment.trainDQN(agent, no_episodes, visualize_agent=visualize_agent_train)

    # Run your agent
    no_episodes_run = 100
    visualize_agent_test = False
    rew_test = environment.runDQN(agent, no_episodes_run, visualize_agent=visualize_agent_test)

    # TODO: Implement here a function visualizing/plotting, e.g.,
    print("-------------------------------------------")
    print("Average Score: {}".format(sum(rew_test) / 100))
    print("-------------------------------------------")
    print("Training Rewards: {}".format(rew_train))
    print("-------------------------------------------")
    print("Testing Rewards: {}".format(rew_test))
    print("-------------------------------------------")
    plotting = True
    if plotting:
        rew_train_without_zeros = [x for x in rew_train if x != 0]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(rew_train_without_zeros)
        ax1.set_title('Training')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Score")
        ax2.plot(rew_test)
        ax2.set_title('Testing')
        ax2.set_xlabel("Episode")
        plt.savefig('Plots.png', dpi=600)
        plt.show()

        # Here you can watch a simulation on how your agent performs after being trained.
        # NOTE that this part will try to load an existing set of weights, therefore set visualize_agent to TRUE, when you
        # already saved a set of weights from a training session
    visualize_agent_afterwards = False
    if visualize_agent_afterwards == True:
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
