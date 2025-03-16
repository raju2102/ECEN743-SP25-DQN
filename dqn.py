"""
ECEN 743: Reinforcement Learning
Deep Q-Learning
Code tested using
1. gymnasium 1.1.1
2. box2d-py  2.3.5
3. pytorch   2.6.0
4. Python    3.11.11

General Instructions
1. This code consists of TODO blocks, read them carefully and complete each of the blocks
2. Type your code between the following lines
            ###### TYPE YOUR CODE HERE ######
            #################################
3. The default hyperparameters should be able to solve LunarLander-v3
4. You do not need to modify the rest of the code for this assignment.
"""

import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import trange
import gymnasium as gym
from pprint import pprint
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo



class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size, gpu_index, replay_sample_type):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))     
        self.batch_size = batch_size
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.replay_sample_type = replay_sample_type


    def add(self, state, action,reward,next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        """
        TODO
        Select indices based on the replay sampling type.
        - 'instant': Take the last 'batch_size' transitions, handling buffer wrap-around.
        - 'experience': Randomly sample 'batch_size' indices from memory.
        """
        ###### TYPE YOUR CODE HERE ######
        #################################

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).long().to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.BoolTensor(self.done[ind]).to(self.device)
        )
    


class EpsilonDecayer:
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def step(self):
        """
        TODO: Write code for decaying the exploration rate using self.epsilon_decay
        and self.epsilon_end. Note that epsilon has been initialized to self.epsilon_start
        """
        ###### TYPE YOUR CODE HERE ######
        #################################



class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        """
        TODO
        1. Define a feedforward neural network with three layers.
        2. The first layer maps the state_dim to 64 hidden units.
        3. The second layer is another hidden layer with 64 units.
        4. The third (output) layer maps to action_dim, producing Q-values for each action.
        """
        ###### TYPE YOUR CODE HERE ######
        #################################
        
    def forward(self, state):
        """
        TODO
        1. Implement the forward pass.
        2. Apply ReLU activation after the first and second layers.
        3. The output layer directly produces Q-values without activation.
        """
        ###### TYPE YOUR CODE HERE ######
        #################################
        return q



class DQNAgent:
    def __init__(self, state_dim, action_dim, discount, tau, lr, update_freq, max_size, batch_size, gpu_index, replay_sample_type, no_target_net, double_dqn, **kwargs):

        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lr = lr
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.no_target_net = no_target_net
        self.double_dqn = double_dqn
        self.t_train = 0

        self.Q = QNetwork(state_dim, action_dim).to(self.device)
        self.Q_target = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

        self.memory = ReplayBuffer(state_dim,1,max_size,self.batch_size,gpu_index,replay_sample_type)
        

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)       
        self.t_train += 1 
                    
        if self.memory.size > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.discount)
        
        if (not self.no_target_net) and ((self.t_train % self.update_freq)==0):
            self.target_update(self.Q, self.Q_target, self.tau)

            
    def select_action(self, state, epsilon):
        """
        TODO
        1. With probability epsilon, select a random action.
        2. Otherwise, use the Q-network to choose the best action.
        """
        ###### TYPE YOUR CODE HERE ######
        ################################# 


    def learn(self, experiences, discount):
        states, actions, rewards, next_states, dones = experiences
        """
        TODO
        1. Compute the target Q-values based on the Bellman equation:
            - If using Double DQN (based on flags):
                - Use the current Q-network to select the best action in `next_states`.
                - Use the target Q-network to evaluate the value of that action.
            - If not using Double DQN (based on flags):
                - Directly compute the maximum Q-value of `next_states` using either the Q-network or target network (based on flags).
        2. Compute the expected Q-values for the chosen actions using the main Q-network.
        3. Compute the loss using Mean Squared Error (MSE) between the expected and target Q-values.
        """
        ###### TYPE YOUR CODE HERE ######
        #################################
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def target_update(self, Q, Q_target, tau):
        """
        TODO
        1. Update the target network parameters (param_target) using current Q parameters (param_Q)
        2. Perform the update using tau, this ensures that we do not change the target network drastically
        3. param_target = tau * param_Q + (1 - tau) * param_target
        """
        ###### TYPE YOUR CODE HERE ######
        #################################



def train_loop(args):
    print('\n\nTraining')
    
    env = gym.make(args.env, render_mode=None)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        **vars(args)
    )
    epsilondecayer = EpsilonDecayer(args.epsilon_start, args.epsilon_end, args.epsilon_decay)

    episode_rewards = []
    episode_pbar = trange(args.n_episodes)
    for _ in episode_pbar:
        state, _ = env.reset(seed=args.seed)
        curr_reward = 0
        for _ in range(args.max_esp_len):
            action = agent.select_action(state, epsilondecayer.epsilon)
            n_state,reward,terminated,truncated,_ = env.step(action)
            done = terminated or truncated
            agent.step(state,action,reward,n_state,done)
            state = n_state
            curr_reward += reward
            if done:
                break
        epsilondecayer.step()
        episode_rewards.append(curr_reward)
        episode_pbar.set_description(f'Episodic Reward {curr_reward:8.2f}')
    env.close()

    return agent, episode_rewards



def plot_rewards(episode_rewards, run_name):
    """
    TODO
    1. Compute a smoothed version of the episodic rewards using a moving average window of size 100.
    2. Plot the original episodic rewards.
    3. Overlay the smoothed reward curve to visualize training progress more clearly.
    4. Label the plot with appropriate titles and axis names.
    5. Save the plot
    """
    ###### TYPE YOUR CODE HERE ######
    #################################



def test_loop(agent, env_name, run_name, max_esp_len):
    print('\n\nTesting')
    env = gym.make(env_name, render_mode="rgb_array")
    """
    TODO
    1. Create a video recording of the agent's performance during testing.
    2. Select an action using the agents policy with the appropriate exploration rate.
    """
    ###### TYPE YOUR CODE HERE ######
    #################################
                


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # main arguments
    parser.add_argument("--run-name", required=True, type=str)
    parser.add_argument("--env", default="LunarLander-v3")                                                      # Gymnasium environment name
    parser.add_argument("--seed", default=0, type=int)                                                          # sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--n-episodes", default=2000, type=int)                                                 # maximum number of training episodes
    parser.add_argument("--batch-size", default=64, type=int)                                                   # training batch size
    parser.add_argument("--discount", default=0.99)                                                             # discount factor
    parser.add_argument("--lr", default=5e-4)                                                                   # learning rate
    parser.add_argument("--tau", default=0.001)                                                                 # soft update of target network
    parser.add_argument("--max-size", default=int(1e5),type=int)                                                # experience replay buffer length
    parser.add_argument("--update-freq", default=4, type=int)                                                   # update frequency of target network
    parser.add_argument("--gpu-index", default=0,type=int)		                                                # GPU index
    parser.add_argument("--max-esp-len", default=1000, type=int)                                                # maximum steps in an episode
    # exploration strategy
    parser.add_argument("--epsilon-start", default=1)                                                           # start value of epsilon
    parser.add_argument("--epsilon-end", default=0.01)                                                          # end value of epsilon
    parser.add_argument("--epsilon-decay", default=0.995)                                                       # decay value of epsilon
    # additional arguments to create variations of q learning algorithm
    parser.add_argument("--replay-sample-type", choices=['instant', 'experience'], default='experience')        # Type of sampling from replay buffer
    parser.add_argument("--no-target-net", action='store_true')                                                 # Flag to learn without target network
    parser.add_argument("--double-dqn", action='store_true')                                                    # Flag to learn with double dqn
    args = parser.parse_args()

    if args.double_dqn:
        assert args.no_target_net is False, "Target Network is required for Double DQN"
    pprint(vars(args), indent=4, width=2)
    os.makedirs(args.run_name, exist_ok=True)

    # setting seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # training agent
    agent, train_episode_rewards = train_loop(args)

    # plotting train rewards
    plot_rewards(train_episode_rewards, args.run_name)

    # testing agent
    test_loop(agent, args.env, args.run_name, args.max_esp_len)