# Run as ./sbatch_gpu_generate_embeddings_mint.sh <job_name> <task>

# Retrieve command name from the command line
JOB_NAME=$1
TASK=$2

mkdir -p logs/slurm

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name=mint_"$JOB_NAME"
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/"$JOB_NAME".out 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=a100
#SBATCH --qos=gpu1week
#SBATCH --reservation=schwede

module load CUDA/12.4.0
export PATH=$HOME/mambaforge/bin:$PATH
source activate mint

srun python3 ./mint/embeddings_mint.py --task "${TASK}" --model_name "mint" \
    --devices "0" \
    --checkpoint_path $HOME/projects/gLM/mint/mint/model/mint.ckpt 

EOF

