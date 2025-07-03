# Run as bash sbatch_train_validate.sh ...

CONFIGS_PAR_DIR=$1
OUTPUT_PAR_DIR=$2
JOB_PAR_NAME=$3
REPRESENTATION=$4
BIOLM=$5
MEM=$6

mkdir -p logs/slurm

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name=gLM_${REPRESENTATION}_${BIOLM}_job
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/${REPRESENTATION}_${BIOLM}_job.out 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --mem-per-cpu="$MEM"G
#SBATCH --qos=gpu1day
#SBATCH --reservation=schwede

module load CUDA/12.4.0
export PATH=$HOME/mambaforge/bin:$PATH
source activate gLM11

srun python3 ./gLM/make_train_validate_config.py -c "${CONFIGS_PAR_DIR}" -o "${OUTPUT_PAR_DIR}" \
    -j "${JOB_PAR_NAME}" -r "${REPRESENTATION}" -b "${BIOLM}" -f 5 --hyperparam "optimizer batch_size"

srun python main.py fit -c ${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/${REPRESENTATION}/${BIOLM}/train_validate/base.yaml

EOF
