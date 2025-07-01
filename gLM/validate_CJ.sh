#!/bin/bash

# Automated pipeline for validation of CJ-based PPI models (all LMs)

# Requirements: 
#   - prepare the parent base.yaml file
#   - generate all representations needed

JOB_PAR_NAME=$1 # parent job name
DEV_SUBSET="validate"
DATA_PAR_DIR="./data/"
CONFIGS_PAR_DIR="./configs/"

# The grid jobs:
# n: 3, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125
# biolm: gLM2, ESM2, MINT

declare -A biolm_model
biolm_model["gLM2"]="gLM2_650M"
biolm_model["ESM2"]="esm2_t33_650M_UR50D"
biolm_model["MINT"]="mint"

declare -A concat_type
concat_type["gLM2"]="gLM2"
concat_type["ESM2"]="pLM"
concat_type["MINT"]="pLM"

declare -A model_path
model_path["gLM2"]="./gLM2_650M/"
model_path["ESM2"]="./esm2_t33_650M_UR50D/"
model_path["MINT"]="./mint/mint/model/mint.ckpt"

declare -A matrix_path
matrix_path["gLM2"]="./outputs/categorical_jacobians/glm2_cosine_post250521/"
matrix_path["ESM2"]="./outputs/categorical_jacobians/esm2_cosine_post250619/"
matrix_path["MINT"]="./outputs/categorical_jacobians/mint_cosine/"

representation="cosine_fast_categorical_jacobian"

for n in 3 2 1 0.5 0.25 0.125 0.0625 0.03125; do
    for biolm in "gLM2" "ESM2" "MINT"; do
        # Making directory and the specific job configuration file
        CONFIGS_DIR="${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/${representation}/${DEV_SUBSET}/n_${n}/"
        mkdir -p "${CONFIGS_DIR}"
            
        sed "s|JOB_PAR_NAME|${JOB_PAR_NAME}|g" "${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/${representation}/base.yaml" |\
            sed "s|model_path: MODEL_PATH|model_path: ${model_path["${biolm}"]}|" |\
            sed "s|matrix_path: MATRIX_PATH|matrix_path: ${matrix_path["${biolm}"]}|" |\
            sed "s|N_THRES|$n|" |\
            sed "s|DEV_SPLIT|${DEV_SUBSET}|" |\
            sed "s|REPRESENTATION|${representation}|g" | sed "s|BIOLM_MODEL|${biolm_model["${biolm}"]}|g" |\
            sed "s|BIOLM|${biolm}|g" | \
            sed "s|concat_type: CONCAT|concat_type: ${concat_type["$biolm"]}|" > "${CONFIGS_DIR}/${biolm}.yaml"

        bash gLM/sbatch_validate_CJ.sh "${JOB_PAR_NAME}/${representation}/${DEV_SUBSET}/n_${n}/${biolm}" \
            "${CONFIGS_DIR}"/${biolm}.yaml 10
        
    done
done

