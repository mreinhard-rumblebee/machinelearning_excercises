import tensorflow as tf
from tensorflow import keras
from keras import layers
import gym
import numpy as np
import scipy.signal
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


class GymEnvironment:
    def __init__(self, env_id, monitor_dir, max_timesteps=400):  # TODO: default max_timesteps = 400
        self.max_timesteps = max_timesteps

        self.env = gym.make(env_id)
        # TODO: Optional: change environment timesteps beyond default of 200 to increase performance beyond 400 timesteps
        # self.env._max_episode_steps = self.max_timesteps

    def trainPPO(self, agent, no_episodes, visualize_agent=False):
        rew = self.runPPO(agent, no_episodes, training=True, visualize_agent=visualize_agent)

        # automatically save current model
        agent.actor.save_weights('cartpole_v0_PPO_actor.h5', overwrite=True)
        agent.critic.save_weights('cartpole_v0_PPO_critic.h5', overwrite=True)

        return rew

    def runPPO(self, agent, no_episodes, training=False, visualize_agent=False):

        rew_total = []
        for episode in range(no_episodes):
            states = np.zeros((agent.actors * self.max_timesteps, agent.state_size), dtype=np.float32)
            actions = np.zeros(agent.actors * self.max_timesteps, dtype=np.int32)
            logprobs = np.zeros(agent.actors * self.max_timesteps, dtype=np.float32)
            advantages = []
            G_t = []
            storage_counter = 0
            rew_episode = []

            for n in range(0, agent.actors):
                state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
                rew_agent = 0
                values = np.zeros(self.max_timesteps)
                rewards = np.zeros(self.max_timesteps + 1)
                for t in range(self.max_timesteps):

                    if visualize_agent:
                        self.env.render()

                    logit, action = agent.select_action(state)
                    next_state, reward, done, _ = self.env.step(action.numpy()[0])

                    next_state = next_state.reshape(1, self.env.observation_space.shape[0])
                    rew_agent += reward

                    if training == True:
                        states[storage_counter] = state
                        actions[storage_counter] = action

                        rewards[t] = reward
                        values[t] = agent.critic(state)[0]
                        logprobs[storage_counter] = policy_probabilities(logit, action)

                    state = next_state
                    storage_counter += 1

                    if ((done == True) or (t == self.max_timesteps - 1)) and (training == True):
                        # Calculate advantages when the function breaks or the last iteration is reached
                        adv, G_t_local = agent.calc_advantage(state, rewards, values, done, t)
                        advantages = np.append(advantages, adv)
                        G_t = np.append(G_t, G_t_local)
                        # print(f'Episode {episode} actor {n}: reward: {rew_agent} /{self.max_timesteps}')
                        break

                rew_episode.append(rew_agent)

            print(
                f'Episode {episode} of {no_episodes} finished with reward: {int(sum(rew_episode) / agent.actors)}/{self.max_timesteps}')
            rew_total.append(int(sum(rew_episode) / agent.actors))

            if training == True:
                # convert to tf tensors
                G_t = tf.convert_to_tensor(G_t, dtype=tf.float32)
                advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
                advantages = (advantages - np.mean(advantages)) / np.std(advantages)

                # truncate to length of G_t
                states = states[0:len(G_t)]
                actions = actions[0:len(G_t)]
                logprobs = logprobs[0:len(G_t)]

                # convert to tf tensors
                states = tf.convert_to_tensor(states)
                actions = tf.convert_to_tensor(actions)
                logprobs = tf.convert_to_tensor(logprobs)

                # update policy and value parameters
                agent.update_policy_parameters(states, actions, logprobs, advantages)
                agent.update_value_parameters(G_t, states)
        if training:
            print('Training done.')
        print(
            f'Total reward after all {no_episodes} episodes: {int(sum(rew_total) / no_episodes)}/{self.max_timesteps}')
        return rew_total


def policy_probabilities(logit, action):
    logprobs = tf.nn.log_softmax(logit)
    logprob = tf.reduce_sum(tf.one_hot(action, 2) * logprobs, axis=1)
    return logprob


# Sum of discountated rewards of vectors --> useful for advantage estimates or total rewards
def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPO_Agent:
    def __init__(self, no_of_states, no_of_actions, load_old_model=False):
        self.state_size = no_of_states
        self.action_size = no_of_actions

        # TODO: Set hyperparameters and vary them
        self.clip_ratio = 0.2  # clipping rate, default 0.2
        self.gamma = 0.99  # discount rate, default 0.99
        self.lr = 0.001  # learning rate, (0.1, 0.001, 0.0003, 0.0001)
        self.lam = 0.95  # lambda for TD(lambda), (0.99, 0.97, 0.95)
        self.actors = 25  # number of actors, (50, 25)

        self.policy_iterations = 20  # number of policy updates
        self.value_iterations = 20  # number of value updates

        # initialize NNs
        self.actor = self.nn_model(self.state_size, self.action_size)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.critic = self.nn_model(self.state_size, 1)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.lr)

    def select_action(self, state):
        logit = agent.actor(state)
        action = tf.squeeze(tf.random.categorical(logit, 1), axis=1)
        return logit, action

    def calc_advantage(self, state, rewards, values, done, t):
        # Using offline forward-looking TD(lambda) with one update per episode
        rewards = rewards[:t + 1]
        values = values[:t + 1]
        if done == True:
            values = np.array(np.append(values, 0))
        else:
            values = np.array(np.append(values, agent.critic(state)[0]))
        TD_delta = rewards + agent.gamma * values[1:] - values[:-1]
        adv = discounted_cumulative_sums(TD_delta, agent.gamma * agent.lam)

        if done == False:
            rewards = np.array(np.append(rewards, agent.critic(state)[0]))
            G_t_local = discounted_cumulative_sums(rewards, agent.gamma)[:-1]
        else:
            G_t_local = discounted_cumulative_sums(rewards, agent.gamma)

        return adv, G_t_local

    def nn_model(self, state_size, output_size, load_old_model=False):
        # If you already have a set of weights from previous training, you can load them here
        if load_old_model:
            if output_size == 1:
                model = self.critic.load_weights("cartpole_v0_PPO_critic.h5")
            else:
                model = self.actor.load_weights("cartpole_v0_PPO_actor.h5")
            return model

        # TODO: vary activation function (relu, tanh, selu)
        input_layer = layers.Input(shape=(state_size,))
        layer_1 = layers.Dense(128, activation="selu")(input_layer)
        layer_2 = layers.Dense(128, activation="selu")(layer_1)
        output_layer = layers.Dense(output_size, activation="linear")(layer_2)
        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model

    # @tf.function
    def update_policy_parameters(self, states, actions, logprobs, advantages):

        for _ in range(self.policy_iterations):
            with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
                ratio = tf.exp(policy_probabilities(agent.actor(states), actions) - logprobs)
                clip = keras.backend.clip(ratio, min_value=1 - self.clip_ratio,
                                          max_value=1 + self.clip_ratio) * advantages
                pol_loss = -keras.backend.mean(tf.minimum(ratio * advantages, clip))

            pol_grads = tape.gradient(pol_loss, agent.actor.trainable_variables)
            agent.actor_optimizer.apply_gradients(zip(pol_grads, agent.actor.trainable_variables))

    # @tf.function
    def update_value_parameters(self, G_t, states):

        for _ in range(self.value_iterations):
            with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
                val_loss = tf.keras.metrics.mean_squared_error(G_t, agent.critic(states))
            val_grads = tape.gradient(val_loss, agent.critic.trainable_variables)
            agent.critic_optimizer.apply_gradients(zip(val_grads, agent.critic.trainable_variables))


if __name__ == "__main__":
    environment = GymEnvironment('CartPole-v0', 'gymresults/cartpole-v0')

    no_of_states = 4  # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    no_of_actions = 2  # [left, right]

    # load weights of an already trained model
    load_old_model = False

    # render agent visualization during training/testing
    visualize_agent_train = False
    visualize_agent_test = False

    # The agent is initialized
    agent = PPO_Agent(no_of_states, no_of_actions, load_old_model=load_old_model)
    no_train_agents = agent.actors

    # Train your agent
    no_episodes = 100  # TODO: Play around with this number, default: 100, testing: 50
    rew_train = environment.trainPPO(agent, no_episodes, visualize_agent=visualize_agent_train)

    # Run your agent
    no_episodes_run = 100  # TODO: Play around with this number, default: 100, testing: 10
    # This is set to one here as multiple actors are only required for training
    agent.actors = 1
    no_test_agents = agent.actors
    rew_test = environment.runPPO(agent, no_episodes_run, visualize_agent=visualize_agent_test)


    # function for creation of plot info-box
    def box_string(no_agents, no_episodes, rew):
        textstr = '\n'.join((
            f'# agents: {no_agents}',
            f'# episodes: {no_episodes}',
            f'avg_reward: {round(np.mean(rew), 2)}',
            f'std_reward: {round(np.std(rew), 2)}'))

        return textstr


    # function for plotting subplots
    def plotting(x=None, ax=None, title='', xlabel='', ylabel='', box_content='', box_props=None):
        if ax is None:
            ax = plt.gca()
        line, = ax.plot(x)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(.56, .14, box_content, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=box_props)

        return line


    # plot training and testing rewards next to each other
    props = dict(boxstyle='round', facecolor='lightsteelblue', alpha=.7)

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), sharey=True)
    plotting(rew_train, ax=ax1, title='Training', xlabel='episode', ylabel='reward',
             box_content=box_string(no_train_agents, no_episodes, rew_train), box_props=props)
    plotting(rew_test, ax=ax2, title='Testing', xlabel='episode',
             box_content=box_string(no_test_agents, no_episodes_run, rew_test), box_props=props)
    plt.tight_layout()
    # TODO: Optional: Save image of plot to current working directory
    # plt.savefig('PPO_eval_plot_1.png', dpi=600)
    plt.show()
