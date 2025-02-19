# Run as ./sbatch_cpu.sh <job_name> <config>

# Retrieve command name from the command line
JOB_NAME=$1
CONFIG_FILE=$2

mkdir -p logs/slurm

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name=gLM_"$JOB_NAME"
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/"$JOB_NAME".out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# activate conda env
export PATH=/scicore/home/schwede/pudziu0000/mambaforge/bin:$PATH
source activate gLM11

srun python main.py fit -c $CONFIG_FILE

EOF
