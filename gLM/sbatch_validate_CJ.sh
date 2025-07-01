# Run as sbatch_cpu_test.sh ...

# Retrieve command name from the command line
JOB_NAME=$1
CONFIG=$2
MEM=$3

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name="$JOB_NAME"
#SBATCH --output=logs/slurm/"$JOB_NAME".out
#SBATCH --cpus-per-task=1
#SBATCH --mem="$MEM"G
#SBATCH --reservation=schwede

# activate conda env
export PATH=/scicore/home/schwede/pudziu0000/mambaforge/bin:$PATH
source activate gLM11

srun python main.py test -c $CONFIG

EOF
