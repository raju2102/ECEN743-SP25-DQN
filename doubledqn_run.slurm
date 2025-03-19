#!/bin/bash

#SBATCH --job-name=RunDQN
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=8G
#SBATCH --output=%j.log

ml purge
ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
source ~/.bashrc
source activate rl_env

cd $SCRATCH/projects/ECEN743-SP25-DQN

echo "Starting Main!!!"
python dqn.py \
--run-name doubledqn \
--tau 0.005 \
--batch-size 32 \
--epsilon-decay 0.999 \
--lr 1e-4 \
--double-dqn
# --no-target-net
# --replay-sample-type experience
# --double-dqn
echo "Done!!!"