#!/bin/bash
#PBS -N simulation_job
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q instructional
#PBS -j oe
#PBS -o simulation.out

# Always change to the directory from which qsub/mpprun was called
cd "$PBS_O_WORKDIR"

# Optional: load CUDA module if needed by your cluster
# module load cuda/11.8

# Run the compiled binary
./simulation
