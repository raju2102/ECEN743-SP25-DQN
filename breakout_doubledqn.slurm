#!/bin/bash

#SBATCH --job-name=RunDQN
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --output=%j.log

ml purge
ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
source ~/.bashrc
source activate rl_env

cd $SCRATCH/projects/ECEN743-SP25-DQN

echo "Starting Main!!!"
python3 dqn_breakout.py \
--run-name doubledqn \
--tau 0.01 \
--epsilon-decay 0.999 \
--lr 2.5e-4 \
--env "ALE/Breakout-v5" \
--double-dqn \
--n-episodes 20000 \
--epoch-offset 27000 \
--epsilon-start 0.1 \
--epsilon-end 0.05
# --no-target-net
# --replay-sample-type experience
# --double-dqn
echo "Done!!!"