# Run as ./sbatch_gpu_test.sh <job_name> <config_file>

# Retrieve command name from the command line
JOB_NAME=$1
CONFIG_FILE=$2
MEM=$3
PARTITION=$4

mkdir -p logs/slurm

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name=gLM_"$JOB_NAME"
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/"$JOB_NAME".out 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu="$MEM"G
#SBATCH --partition="$PARTITION"
#SBATCH --qos=gpu6hours
#SBATCH --reservation=schwede

module load CUDA/12.4.0
export PATH=$HOME/mambaforge/bin:$PATH
source activate gLM11

srun python main.py test -c $CONFIG_FILE

EOF
