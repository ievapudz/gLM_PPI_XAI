#!/bin/bash

# Automated pipeline for cross-validation of PPI models (all LMs)

# Requirements: 
#   - prepare the parent base.yaml file
#   - generate all representations needed

JOB_PAR_NAME=$1 # parent job name
DEV_SUBSET="CV"
DATA_PAR_DIR="./data/"
CONFIGS_PAR_DIR="./configs/"

# The grid jobs:
# joint_pooling, separate_pooling, joint_input_separate_pooling
# gLM2, ESM2, MINT

declare -A biolm_model
biolm_model["gLM2"]="gLM2_650M"
biolm_model["ESM2"]="esm2_t33_650M_UR50D"
biolm_model["MINT"]="mint"

declare -A concat_type
concat_type["gLM2"]="gLM2"
concat_type["ESM2"]="pLM"
concat_type["MINT"]="pLM"

declare -A emb_dims
emb_dims["joint_pooling"]="1280"
emb_dims["separate_pooling"]="2560"
emb_dims["joint_input_separate_pooling"]="2560"

for representation in "joint_pooling" "separate_pooling" "joint_input_separate_pooling"; do
    for biolm in "gLM2" "ESM2" "MINT"; do
        if [[ "$representation" == joint_input_separate_pooling && "$biolm" == MINT ]]; then
            echo "Skipping $representation for $biolm..."
        else
            # Making directory and the specific job configuration file
            CONFIGS_CV_DIR="${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/${representation}/${biolm}/${DEV_SUBSET}/"
            mkdir -p "${CONFIGS_CV_DIR}"
            
            sed "s|JOB_PAR_NAME|${JOB_PAR_NAME}|g" "${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/base.yaml" |\
                sed "s|REPRESENTATION|${representation}|g" | sed "s|BIOLM_MODEL|${biolm_model["${biolm}"]}|g" |\
                sed "s|BIOLM|${biolm}|g" | sed "s|prefix: 'CV'|prefix: ${DEV_SUBSET}|g" | \
                sed "s|emb_dim: 1280|emb_dim: ${emb_dims["$representation"]}|" | \
                sed "s|concat_type: CONCAT|concat_type: ${concat_type["$biolm"]}|" > "${CONFIGS_CV_DIR}/base.yaml"
            
            # Making configuration files for CV
            bash ./gLM/make_cv_config.sh "${CONFIGS_CV_DIR}" ""

            # Run the CV
            for i in 0 1 2 3 4; do
                bash sh_scripts/sbatch_gpu.sh "${JOB_PAR_NAME}/${representation}/${biolm}/${DEV_SUBSET}/$i" \
                    "${CONFIGS_CV_DIR}"/"$i".yaml "" 2
            done
        fi
    done
done

