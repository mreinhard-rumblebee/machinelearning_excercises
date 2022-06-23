from keras.models import Sequential, Model
from keras.layers import Dense, Input
import tensorflow as tf
from keras import backend as K
import gym
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


class GymEnvironment:
    def __init__(self, env_id, monitor_dir, max_timesteps=400):
        self.max_timesteps = max_timesteps

        self.env = gym.make(env_id)

    def trainPPO(self, agent, no_episodes):
        self.runPPO(agent, no_episodes, training=True)

    def runPPO(self, agent, no_episodes, training=False):

        rew = np.zeros(no_episodes)
        for episode in range(no_episodes):
            state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
            states, actions, G_lams, values_global, logprobs = [], [], [], [], []

            for n in range(0, self.actors):
                state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
                # Problem: if done==1, the pole will always fall, so a reset might be good ???
                tot_rew = 0
                values, rewards = [], []
                for t in range(self.max_timesteps):
                    # TODO: Fill out the respective to-dos in this loop and make sure that the overall algorithm works,
                    #  e.g., overwrite current state with next state entering a new time step

                    logit, action = agent.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = next_state.reshape(1, self.env.observation_space.shape[0])
                    tot_rew += reward

                    states.append(state)
                    actions.append(action)

                    if training == True:
                        # TODO: Store relevant transition information such as rewards, values, etc. that you will need in
                        #  the calculation of the advantages later
                        rewards.append(reward)
                        values.append(agent.critic.predict(state)[0])
                        logprobs.append(policy_probabilities(logit, action))

                    state = next_state

                    if (done == True or t == self.max_timesteps) and training == True:
                        # TODO: Call function for calculation and storage of advantages
                        # Calculate advantages when the function breaks or the last iteration is reached
                        G_lams.extend(agent.calc_advantage(values, rewards, 0, t))
                        values_global.extend(values)

                        gradients.append()
                        # TODO: Store targets for your value function update

                        break
                rew.append(tot_rew)

            # TODO: If training, call function to update policy function weights using clipping
            # TODO: If training, Call function to update value function weights

            # TODO: Implement here a function that evaulates the agent's performance for every x episodes by -- NOT YET
            # calling runDQN directly and returns an average of total rewards for 100 runs, if your objective is
            # reached, you can terminate training
        return rew


def policy_probabilities(logit, action):
    # TODO: Compute probabilities of taking actions a by using the outputs of actor NN (the logits)
    # softmax calculation
    logprobs = tf.math.log(np.exp(logit) / np.sum(np.exp(logit)))
    logprob = tf.reduce_sum(tf.one_hot(action, 2) * logprobs, axis=1)
    return logprob


# Sum of discountated rewards of vectors --> useful for advantage estimates or total rewards
def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPO_Agent:
    def __init__(self, no_of_states, no_of_actions):
        self.state_size = no_of_states
        self.action_size = no_of_actions

        # TODO: Set hyperparameters and vary them
        self.gamma = 0.9  # discount rate
        self.lam = 0.6  # lambda for TD(lambda)
        self.clip_ratio = 0.5  # Clipping ratio for calculating L_clip
        self.actors = 100  # Number of parallel actors

        self.actor = self.nn_model(self.state_size, self.action_size)
        self.critic = self.nn_model(self.state_size, 1)

    def select_action(self, state):
        # TODO: Implement action selection, i.e., sample an action from policy pi
        logit = self.actor.predict(state)[0]
        action = tf.squeeze(tf.random.categorical(logit, 1), axis=1)
        return logit, action

    def calc_advantage(self, values, rew, n, T):
        # Using offline forward-looking TD(lambda) with one update per episode
        G_lam = []
        for t in range(T):
            acc_rew = 0
            rew_list = []
            for k in range(T - t):
                acc_rew = acc_rew + self.gamma ** k * rew[t + k]
                V_tmp = self.gamma ** (k + 1) * values[t + k + 1] if k < T - t - 1 else 0
                rew_list.append((acc_rew + V_tmp) * self.lam ** k)

            G_t = (1 - self.lam) * np.sum(rew_list[:-1]) + rew_list[-1]

            G_lam.append(G_t)

            """
            G_n = np.zeros(T-t+1)
            for k in range(T-t+1):
                rew_disc = 0
                for i in range(k):
                    rew_disc = rew_disc + rew[t+i] * self.gamma**i
                G_n[k] = rew_disc + values[T+k+1] * self.gamma**(k+1)
            for i in range(t-(T-t)):
                G_n[T-t] = G_n[T-t] + rew[t+i] * self.gamma**i
            G_disc = 0
            for k in range(1, T-t-1):
                G_disc = G_disc + G_n[k-1] * self.lam**(k-1)
            G_lam = (1-self.lam) * G_disc + G_n[T-t] * self.lam**(T-t-1)
            """

            return G_lam

        # Consider normalizing the advantages:
        # TD = (TD - np.mean(TD)) / (np.std(TD) + 1e-10)

        # TODO: Implement here the calculation of the advantage, e.g., using TD-lambda or eligibility traces

    def nn_model(self, state_size, output_size):
        layer_input = Input(state_size)
        layer_one = Dense(512, input_shape=state_size, activation="relu", kernel_initializer='random_normal',
                          bias_initializer='zeros')(layer_input)
        layer_two = Dense(512, activation="relu", kernel_initializer='random_normal', bias_initializer='zeros')(
            layer_one)
        layer_three = Dense(512, activation="relu", kernel_initializer='random_normal', bias_initializer='zeros')(
            layer_two)
        layer_four = Dense(output_size, activation="linear", kernel_initializer='random_normal',
                           bias_initializer='zeros')(layer_three)

        model = Model(inputs=layer_input, outputs=layer_four)
        model.compile(loss="mse", metrics=["accuracy"])
        print(model)
        # TODO: Define the neural network here, make sure that you account for the different requirements of the value
        # function and policy function approximation in the in- and outputs

    # Here newly observed transitions are stored in the experience replay buffer
    def record(self):  # TODO: add the relevant input arguments that you will need to store

    # TODO: Define here arrays in which you will store all the information that you need in the advantage
    #  calculation, e.g., rewards, values, states, etc.

    @tf.function  # This is a wrapper that when adding it in front of a function, consisting only of tf syntax,
    # can improve speed
    def update_policy_parameters(self, states, actions, logprobs, G_lams, values_global):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            # TODO: Use the advantages and calculated policies to calculated the clipping function here and calculate
            #  the loss function
            ratio = tf.exp(policy_probabilities(self.actor.predict(states)[0], actions) - logprobs)

            advantages = list(map(lambda x, y: x - y, G_lams, values_global))
            clip = K.clip(ratio, min_value=1 - self.clip_ratio, max_value=1 + self.clip_ratio) * advantages
            pol_loss = -K.mean(K.minimum(ratio * advantages, clip))

        pol_grads = tape.gradient(pol_loss, self.actor.trainable_variables)
        self.actor.apply_gradients(zip(pol_grads, self.actor.trainable_variables))

    # This is a wrapper that when adding it in front of a function, consisting only of tf syntax,
    # can improve speed
    @tf.function
    def update_value_parameters(self, alpha, G_lams, values_global):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            # TODO: Use the advantages and calculated policies to calculated the clipping function here and calculate
            #  the loss function
            val_loss = alpha * tf.keras.metrics.mean_squared_error(G_lams, values_global)

        val_grads = tape.gradient(val_loss, self.critic.trainable_variables)
        self.critic.apply_gradients(zip(val_grads, self.critic.trainable_variables))


if __name__ == "__main__":
    environment = GymEnvironment('CartPole-v0', 'gymresults/cartpole-v0')

    no_of_states = 4  # TODO: Define number of states # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    no_of_actions = 2  # TODO: Define number of actions # [left, right]

    # The agent is initialized
    agent = PPO_Agent(no_of_states, no_of_actions)

    # Train your agent
    no_episodes = 500  # TODO: Play around with this number
    environment.trainPPO(agent, no_episodes)

    # Run your agent
    no_episodes_run = 100
    agent.actors = 1  # This is set to one here as multiple actors are only required for training
    rew = environment.runPPO(agent, no_episodes_run)

    # TODO: Implement here a function visualizing/plotting, e.g., -- NOT YET
    # your agent's performance over the number of training episodes

# Useful Links:
# https://www.youtube.com/watch?v=lYP3cF2wqOY
# https://pylessons.com/PPO-reinforcement-learning
# https://github.com/pythonlessons/Reinforcement_Learning/blob/master/11_Pong-v0_PPO/Pong-v0_PPO_TF2.py
# https://towardsdatascience.com/understanding-and-implementing-proximal-policy-optimization-schulman-et-al-2017-9523078521ce
# https://medium.com/intro-to-artificial-intelligence/the-actor-critic-reinforcement-learning-algorithm-c8095a655c14