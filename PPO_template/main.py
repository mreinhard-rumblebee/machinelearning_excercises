
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
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

            for n in range(0,self.actors):
                tot_rew = 0
                rew = []
                for t in range(self.max_timesteps):
                    # TODO: Fill out the respective to-dos in this loop and make sure that the overall algorithm works,
                    #  e.g., overwrite current state with next state entering a new time step

                    logit,action = agent.select_action(state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = next_state.reshape(1, self.env.observation_space.shape[0])
                    tot_rew += reward


                    if training == True:
                    # TODO: Store relevant transition information such as rewards, values, etc. that you will need in
                    #  the calculation of the advantages later

                    if done and training == True:
                        #TODO: Call function for calculation and storage of advantages
                        #TODO: Store targets for your value function update
                        break
                rew.append(tot_rew)

            #TODO: If training, call function to update policy function weights using clipping
            #TODO: If training, Call function to update value function weights
            # TODO: Implement here a function that evaulates the agent's performance for every x episodes by
            # calling runDQN directly and returns an average of total rewards for 100 runs, if your objective is
            # reached, you can terminate training
        return rew


def policy_probabilities(logit, a):
        #TODO: Compute probabilities of taking actions a by using the outputs of actor NN (the logits)

# Sum of discountated rewards of vectors --> useful for advantage estimates or total rewards
def discounted_cumulative_sums(x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPO_Agent:
    def __init__(self, no_of_states, no_of_actions):
        self.state_size = no_of_states
        self.action_size = no_of_actions

        #TODO: Set hyperparameters and vary them
        self.gamma =   # discount rate
        self.lam =  # lambda for TD(lambda)
        self.clip_ratio =  # Clipping ratio for calculating L_clip
        self.actors =  # Number of parallel actors

        self.actor = self.nn_model(self.state_size,self.action_size)
        self.critic =self.nn_model(self.state_size,1)


    def select_action(self,state):
        #TODO: Implement action selection, i.e., sample an action from policy pi

    def calc_advantage(self,values,rew,n,T):
        #TODO: Implement here the calculation of the advantage, e.g., using TD-lambda or eligibility traces

    def nn_model(self,state_size,output_size):
        #TODO: Define the neural network here, make sure that you account for the different requirements of the value
        # function and policy function approximation in the in- and outputs

    # Here newly observed transitions are stored in the experience replay buffer
    def record(self): # TODO: add the relevant input arguments that you will need to store
        # TODO: Define here arrays in which you will store all the information that you need in the advantage
        #  calculation, e.g., rewards, values, states, etc.

    @tf.function # This is a wrapper that when adding it in front of a function, consisting only of tf syntax,
    # can improve speed
    def update_policy_parameters(self):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            # TODO: Use the advantages and calculated policies to calculated the clipping function here and calculate
            #  the loss function
            pol_loss =

        pol_grads = tape.gradient(pol_loss, self.actor.trainable_variables)
        self.actor.apply_gradients(zip(pol_grads, self.actor.trainable_variables))

    # This is a wrapper that when adding it in front of a function, consisting only of tf syntax,
    # can improve speed
    @tf.function
    def update_value_parameters(self):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            # TODO: Use the advantages and calculated policies to calculated the clipping function here and calculate
            #  the loss function
            val_loss = ...
        val_grads = tape.gradient(val_loss, self.critic.trainable_variables)
        self.critic.apply_gradients(zip(val_grads, self.critic.trainable_variables))


if __name__ == "__main__":
    environment = GymEnvironment('CartPole-v0', 'gymresults/cartpole-v0')


    no_of_states = #TODO: Define number of states
    no_of_actions = #TODO: Define number of actions

    # The agent is initialized
    agent = PPO_Agent(no_of_states,no_of_actions)

    # Train your agent
    no_episodes = 500 # TODO: Play around with this number
    environment.trainPPO(agent, no_episodes)

    # Run your agent
    no_episodes_run =
    agent.actors = 1 # This is set to one here as multiple actors are only required for training
    rew = environment.runPPO(agent, no_episodes_run)

    # TODO: Implement here a function visualizing/plotting, e.g.,
    # your agent's performance over the number of training episodes
