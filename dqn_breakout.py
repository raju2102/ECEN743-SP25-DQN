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
import gym
from pprint import pprint
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import imageio
import collections
from gym import ObservationWrapper
import cv2
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
from gym.spaces import Box
import collections

class GrayscaleObservation(ObservationWrapper):
    def __init__(self, env):
        super(GrayscaleObservation, self).__init__(env)
        self.observation_space = Box(low=0.0, high=1.0, shape=(84, 84), dtype=np.float32)

    def observation(self, obs):
        if obs.shape[-1] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return (gray / 255.0).astype(np.float32)

class FireResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FireResetWrapper, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE', "Environment does not support FIRE action!"
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs) 
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        return obs, info

class FrameStackRepeatAction(ObservationWrapper):
    def __init__(self, env, n_frames):
        super(FrameStackRepeatAction, self).__init__(env)
        self.n_frames = n_frames
        self.observation_space = Box(
            low=0.0, high=1.0, 
            shape=(n_frames, *env.observation_space.shape[:2]), 
            dtype=np.float32
        )
        self.frame_buffer = collections.deque(maxlen=n_frames)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frame_buffer.append(obs)
        return self._get_stacked_frames(), info

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.n_frames):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.frame_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                done = True
                break
        
        return self._get_stacked_frames(), total_reward, terminated, truncated, info

    def _get_stacked_frames(self):
        return np.stack(self.frame_buffer, axis=0)



class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size, gpu_index, replay_sample_type):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))     
        self.batch_size = batch_size
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device("mps")
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
        batch_size = self.batch_size 
        buffer_size = self.size  

        if self.replay_sample_type == "instant":
            ind = np.arange(max(0, buffer_size - batch_size), buffer_size)
        elif self.replay_sample_type == "experience":
            ind = np.random.choice(buffer_size, min(batch_size, buffer_size), replace=False)

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
        # self.epsilon = self.epsilon + (self.epsilon_end - self.epsilon) * np.exp(-self.epsilon_decay)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        ###### TYPE YOUR CODE HERE ######
        #################################



class QNetwork(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(QNetwork, self).__init__()
        # print(input_shape)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
        
    


class DQNAgent:
    def __init__(self, state_dim, action_dim, discount, tau, lr, update_freq, max_size, batch_size, gpu_index, replay_sample_type, no_target_net, double_dqn, **kwargs):

        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lr = lr
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device("mps")
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
        # if np.random.rand() > epsilon:
        #     return self.Q(state).cpu().numpy().argmax()
        # else:
        #     return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if np.random.rand() > epsilon:
            with torch.no_grad():
                return self.Q(state_tensor).cpu().numpy().argmax() 
        else:
            return np.random.randint(self.action_dim)
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
        if self.double_dqn:
            best_actions = self.Q(next_states).argmax(dim=1, keepdim=True)
            max_Q = self.Q_target(next_states).gather(1, best_actions).squeeze(1)
        else:
            max_Q = self.Q_target(next_states).max(dim=1)[0]

        target_Q = rewards + discount * max_Q.unsqueeze(1) * (1 - dones.float())
        # expected_Q = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        expected_Q = self.Q(states).gather(1, actions.view(-1, 1))
        loss = nn.MSELoss()(expected_Q, target_Q)
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
        for param_Q, param_target in zip(Q.parameters(), Q_target.parameters()):
            param_target.data.copy_(tau * param_Q.data + (1 - tau) * param_target.data)
        ###### TYPE YOUR CODE HERE ######
        #################################



def train_loop(args):
    print('\n\nTraining')
    env = gym.make(args.env, render_mode="rgb_array")
    env = FireResetWrapper(env)
    env = GrayscaleObservation(env)              
    env = FrameStackRepeatAction(env, 4)         
    print(env.observation_space.shape) 
    agent = DQNAgent(
        state_dim=env.observation_space.shape,
        action_dim=env.action_space.n,
        **vars(args)
    )
    epsilondecayer = EpsilonDecayer(args.epsilon_start, args.epsilon_end, args.epsilon_decay)
    offset = 0
    if args.epoch_offset:
        try:
            offset = int(args.epoch_offset)
            checkpoint_path = os.path.join(args.run_name, f"checkpoint_ep{offset}.pth")
            agent.Q.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    episode_rewards = []
    episode_pbar = trange(args.n_episodes)
    for ep in episode_pbar:
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
        if (ep + 1) % 1000 == 0:
            checkpoint_ep = offset + ep + 1
            checkpoint_path = os.path.join(args.run_name, f"checkpoint_ep{checkpoint_ep}.pth")
            torch.save(agent.Q.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at episode {checkpoint_ep} : {checkpoint_path}")
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
    rewards = np.array(episode_rewards)
    window_size = 100
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Original Rewards", alpha=0.3)
    plt.plot(range(window_size - 1, len(rewards)), smoothed_rewards, label="Smoothed Rewards (100 episodes)", color='red')

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(f"Training Progress - {run_name}")
    plt.legend()
    plt.grid(True)

    filename = f"{run_name}_rewards_plot.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()
    ###### TYPE YOUR CODE HERE ######
    #################################



def test_loop(agent, env_name, run_name, max_esp_len):
    print('\n\nTesting')

    env = gym.make(env_name, render_mode="rgb_array")
    env = FireResetWrapper(env)
    env = GrayscaleObservation(env)
    env = FrameStackRepeatAction(env, n_frames=4)

    best_reward = float('-inf')
    best_frames = []

    for attempt in range(10):
        print(f"Test Attempt {attempt + 1}/10")
        state, _ = env.reset()
        frames = []
        total_reward = 0

        for t in range(max_esp_len):
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            frames.append(env.render()) 
            state = next_state
            if done or truncated:
                break

        print(f"Total Reward for Attempt {attempt + 1}: {total_reward}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames

        if total_reward > 100:  
            print("Successful episode achieved!")
            break

    video_filename = f"{run_name}_best_test_run.mp4"
    imageio.mimsave(video_filename, best_frames, fps=30)
    
    print(f"Best test run completed with Total Reward: {best_reward}")
    print(f"Video saved as {video_filename}")

    env.close()
    ###### TYPE YOUR CODE HERE ######
    #################################
                


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # main arguments
    parser.add_argument("--run-name", required=True, type=str)
    parser.add_argument("--env", default="ALE/Breakout-v5")                                                      # Gymnasium environment name
    parser.add_argument("--seed", default=0, type=int)                                                          # sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--n-episodes", default=2000, type=int)                                                 # maximum number of training episodes
    parser.add_argument("--batch-size", default=64, type=int)                                                   # training batch size
    parser.add_argument("--discount", default=0.99)                                                             # discount factor
    parser.add_argument("--lr", default=5e-4, type=float)                                                                   # learning rate
    parser.add_argument("--tau", default=0.001, type=float)                                                                 # soft update of target network
    parser.add_argument("--max-size", default=int(1e5),type=int)                                                # experience replay buffer length
    parser.add_argument("--update-freq", default=4, type=int)                                                   # update frequency of target network
    parser.add_argument("--gpu-index", default=0,type=int)		                                                # GPU index
    parser.add_argument("--max-esp-len", default=1000, type=int)                                                # maximum steps in an episode
    # exploration strategy
    parser.add_argument("--epsilon-start", default=1)                                                           # start value of epsilon
    parser.add_argument("--epsilon-end", default=0.01)                                                          # end value of epsilon
    parser.add_argument("--epsilon-decay", default=0.995, type=float)                                                       # decay value of epsilon
    # additional arguments to create variations of q learning algorithm
    parser.add_argument("--replay-sample-type", choices=['instant', 'experience'], default='experience')        # Type of sampling from replay buffer
    parser.add_argument("--no-target-net", action='store_true')                                                 # Flag to learn without target network
    parser.add_argument("--double-dqn", action='store_true')                                                    # Flag to learn with double dqn
    parser.add_argument("--epoch-offset", required=False, type=str)
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