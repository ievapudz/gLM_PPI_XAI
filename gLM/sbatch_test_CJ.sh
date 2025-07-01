# Run as bash sbatch_test_CJ.sh ...

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
#SBATCH --output=logs/slurm/representations_${BIOLM}.out 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem="${MEM}"G
#SBATCH --qos=30min
#SBATCH --reservation=schwede

module load CUDA/12.4.0
export PATH=$HOME/mambaforge/bin:$PATH
source activate gLM11

srun python3 ./gLM/make_test_cj_config.py -c "${CONFIGS_PAR_DIR}" -o "${OUTPUT_PAR_DIR}" \
    -j "${JOB_PAR_NAME}" -r "${REPRESENTATION}" -b "${BIOLM}" --hyperparam "n"

srun python main.py test -c ${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/${REPRESENTATION}/${BIOLM}/test/base.yaml

EOF
