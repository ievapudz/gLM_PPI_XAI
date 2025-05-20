# Run as ./sbatch_cpu.sh <job_name> <config>

# Retrieve command name from the command line
JOB_NAME=$1
SPLIT=$2
BIOLM=$3

mkdir -p logs/slurm

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name=contact_map_"$JOB_NAME"
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/"$JOB_NAME".out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=6hours

# activate conda env
export PATH=/scicore/home/schwede/pudziu0000/mambaforge/bin:$PATH
source activate glam

srun python3 data_process/contact_map.py -s $SPLIT -b $BIOLM

EOF
