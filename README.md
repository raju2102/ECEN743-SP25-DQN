

# ECEN743-SP25-DQN

## Overview

1. You have to submit a PDF report, your code, and a video to Canvas.
2. Put all your files (PDF report, code, and video) into a **single compressed folder** named `Lastname_Firstname_A4.zip`.
3. Your PDF report should include answers and plots to all the questions.

## General Instructions

1.  This assignment must be completed on the TAMU HPRC. Apply for an account [here](https://hprc.tamu.edu/).
1.  You will complete this assignment in a Python (.py) file. Please check `dqn.py` for the starter code.
1.  Type your code between the following lines
    ```
    ###### TYPE YOUR CODE HERE ######
    #################################
    ```
1. You do not need to modify the rest of the code for this assignment. The default hyperparameters should be able to solve LunarLander-v3.
1. The problem is solved if the **total reward per episode** is 200 or above. *Do not* stop training on the first instance your agent gets a reward above 200, and your agent must achieve a reward of  200 or above consistently.
1. The x-axis of your training plots should be  training episodes (or training iterations), and the y-axis should be episodic reward (or average episodic reward per iteration). You may have to use a sliding average window to get clean plots.
1. **Video Generation:** You do not have to write your own method for video generation. Gymnasium has a nice, self-containted wrapper for this purpose. Please read more about Gymnasium wrappers [here](https://gymnasium.farama.org/api/wrappers/).

## Problems

In this homework, you will train a deep Q-Learning algorithm to land a lunar lander on the surface of the moon. We will use the Lunar Lander environment (LunarLander-v3) from  Gymnasium. The environment consists of a lander with $4$ discrete actions and a continuous state space. A detailed description of the environment can be found [here](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

1. **Deep Q-Learning:** Implement deep Q-learning **with** experience replay and target network. You should include the training curve in the PDF report (x-axis is the number of episode and the y-axis should be episodic cumulative reward). Use a sliding window average to get smooth plots. Include a description of the hyperparameters used. Also, you should submit a video of the smooth landing achieved by your RL algorithm. Try to find the optimal hyperparameters that will enable fast convergence to the optimal policy.  

2. **Ablation Study:** Perform ablation study to understand the importance of experience replay and target network. You should include the training curves, and  describe your observations and inferences.   

3. **Algorithmic Improvements:** As we discussed in the class, there has been a number improvements in the deep Q-learning algorithm and architecture, such as double DQN, dueling DQN, prioritized experience replay, and many other (see the slides and related papers).  Implement double DQN. Explain your implementation and demonstrate the benefit.  

4. **Deep Q-Learning on Another Environment:** Congratulations on implementing deep Q-Learning on the Lunar-Lander-v3 environment! Now,  learn the optimal  policy for another control problem (environment) using deep Q-Learning. You can select one environment from the *Classical Control* set, *Box2D* set or *Atari Games* set in Gymnasium.  In the PDF report, you need to clearly specify the environment and provide a link to the corresponding page in Gymnasium. You need to include the training curve and describe the hyperparameters used. You should also include the video of the performance. 

## HPRC Intructions

### Installation
```
cd $SCRATCH
ml GCCcore/13.3.0
ml git/2.45.1
ml Miniconda3/23.10.0-1
conda init
source ~/.bashrc
git clone https://github.com/ECEN743-TAMU/ECEN743-SP25-DQN 
cd ECEN743-SP25-DQN 
conda create -n rl_env python=3.11
source activate rl_env
pip install -r requirements.txt
```

### Running
After making required changes to `dqn.py` and `run.slurm` 
```
conda deactivate
sbatch run.slurm
```