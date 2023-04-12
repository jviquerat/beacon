#!/bin/bash
#
#SBATCH --job-name=drl_shkadov
#SBATCH --output=drl_shkadov.txt
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --ntasks-per-core 1
#SBATCH --threads-per-core 1
#SBATCH --time=24:00:00
#
source /home/jviquerat/scratch/dragonfly/venv/bin/activate
module load openmpi/4.1.1
mpirun -n 1 dgf --eval -net results/shkadov_ppo/0/ppo -json ppo_buffer.json -steps 400
