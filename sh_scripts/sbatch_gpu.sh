# Run as ./sbatch_gpu.sh <job_name> <config_file>

# Retrieve command name from the command line
JOB_NAME=$1
CONFIG_FILE=$2
MESSAGE=$3
MEM=$4

mkdir -p logs/slurm

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name=gLM_"$JOB_NAME"
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/"$JOB_NAME".out 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu="$MEM"G
#SBATCH --partition=a100
#SBATCH --qos=gpu1day
#SBATCH --reservation=schwede

module load CUDA/12.4.0
export PATH=$HOME/mambaforge/bin:$PATH
source activate gLM11

git add gLM/models.py gLM/dataloader.py gLM/callbacks.py 
git commit -m "$MESSAGE"

srun python main.py fit -c $CONFIG_FILE


EOF
