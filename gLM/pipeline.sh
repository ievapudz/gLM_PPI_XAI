#!/bin/bash

# Automated pipeline for cross-validation and testing of PPI models (all LMs)

# Requirements: 
#   - prepare the parent base.yaml file
#   - generate all representations needed

JOB_PAR_NAME=$1 # parent job name
DEV_SUBSET=$2
DATA_PAR_DIR="./data/"
CONFIGS_PAR_DIR="./configs/"

if [[ "$DEV_SUBSET" == CV ]]; then
    echo "Running CV"
elif [[ "$DEV_SUBSET" == test ]]; then
    echo "Running test"
fi

# The grid jobs:
# joint_pooling, separate_pooling, joint_input_separate_pooling
# gLM2, ESM2, MINT

declare -A biolm_model
biolm_model["gLM2"]="gLM2_650M"
biolm_model["ESM2"]="esm2_t33_650M_UR50D"
biolm_model["MINT"]="mint"

for representation in "joint_pooling" "separate_pooling" "joint_input_separate_pooling"; do
    for biolm in "gLM2" "ESM2" "MINT"; do
        if [[ "$representation" == joint_input_separate_pooling && "$biolm" == MINT ]]; then
            # Maybe just skip
            echo "Skipping $representation for $biolm..."
        else
            # TODO: make the directory
            sed "s|JOB_PAR_NAME|${JOB_PAR_NAME}|g" "${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/base.yaml" |\
                sed "s|REPRESENTATION|${representation}|g" | sed "s|BIOLM_MODEL|${biolm_model["${biolm}"]}|g" |\
                sed "s|BIOLM|${biolm}|g" | sed "s|prefix: 'CV'|prefix: ${DEV_SUBSET}|g"
            #bash ./gLM/make_cv_config.sh "${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/${representation}/${biolm}/${DEV_SUBSET}" ""
        fi
    done
done

# TODO: collect the results: plot the metrics, pick the best epoch of the model

# TODO: generate config files for validation (full training + validation set)

# TODO: train fully until the chosen epoch and validate

# If dev. subset == test:

# TODO: generate config files for testing (change the run name, add ckpt_path)

# TODO: retrieve testing (point) metrics
