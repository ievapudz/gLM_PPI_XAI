# Run as bash sbatch_test.sh ...

CONFIGS_PAR_DIR=$1
LOGS_DIR=$2
JOB_PAR_NAME=$3
OUT_JOB_PAR_NAME=$4
REPRESENTATION=$5
BIOLM=$6
MEM=$7

mkdir -p logs/slurm

# Use a heredoc to create the script
cat << EOF | sbatch
#!/bin/bash
#SBATCH --job-name=${OUT_JOB_PAR_NAME}/${REPRESENTATION}/test/${BIOLM}
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/${OUT_JOB_PAR_NAME}/${REPRESENTATION}/test/${BIOLM}.out 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem="${MEM}"G
#SBATCH --qos=6hours
#SBATCH --reservation=schwede

module load CUDA/12.4.0
export PATH=$HOME/mambaforge/bin:$PATH
source activate gLM11

srun python3 ./gLM/make_test_cj_config.py -c "${CONFIGS_PAR_DIR}" -o "${LOGS_DIR}" \
    -j "${JOB_PAR_NAME}" --out-job-name "${OUT_JOB_PAR_NAME}" -r "${REPRESENTATION}" \
    -b "${BIOLM}" --hyperparam "n"

echo "Running ${CONFIGS_PAR_DIR}/${OUT_JOB_PAR_NAME}/${REPRESENTATION}/test/${BIOLM}.yaml"

srun python main.py test -c ${CONFIGS_PAR_DIR}/${OUT_JOB_PAR_NAME}/${REPRESENTATION}/test/${BIOLM}.yaml

EOF
