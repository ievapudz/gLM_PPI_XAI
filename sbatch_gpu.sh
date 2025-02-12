# Run as ./sbatch_gpu.sh <job_name>

# Retrieve command name from the command line
JOB_NAME=$1

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name=gLM_"$JOB_NAME"
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=a100
#SBATCH --qos=gpu30min
#SBATCH --output="$JOB_NAME".out

# activate conda env
export PATH=/scicore/home/schwede/pudziu0000/mambaforge/bin:$PATH
source activate gLM

nvidia-smi
srun python categorical_jacobian_gLM2.py

EOF
