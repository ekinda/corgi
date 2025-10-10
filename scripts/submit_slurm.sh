#!/bin/bash -l
#SBATCH -o ./corgi_train.out.%j
#SBATCH -e ./corgi_train.err.%j
#SBATCH -D ./
#SBATCH -J corgi_train
#SBATCH --constraint="gpu"       # Use GPU nodes
#SBATCH --cpus-per-task=4        # Request 4 CPU cores per task

#SBATCH --nodes=                 # Request x nodes
#SBATCH --ntasks-per-node=       # Launch x tasks per node (one per GPU)
#SBATCH --gres=                  # Request all GPUs on each node
#SBATCH --mem=             
#SBATCH --time=

export MASTER_PORT=12345
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

module purge

. virtualenv/bin/activate
srun python3 -u -Wi main_train.py
