# Run as ./sbatch_cpu.sh <script> <job_name>

# Retrieve command name from the command line
SCRIPT=$1
JOB_NAME=$2

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name="$JOB_NAME"
#SBATCH --output="$JOB_NAME".out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# activate conda env
export PATH=/scicore/home/schwede/pudziu0000/mambaforge/bin:$PATH
source activate gLM11

srun python main.py fit -c configs/config_simple.yaml

EOF
