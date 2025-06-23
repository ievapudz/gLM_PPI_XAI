#!/bin/bash

# Automated pipeline for cross-validation and testing of PPI models (all LMs)

# Requirements: prepare the base.yaml file

JOB_PAR_NAME=$1 # parent job name
DEV_SUBSET=$2
DATA_PAR_DIR="./data/"
CONFIGS_PAR_DIR="./configs/"

if [[ "$DEV_SUBSET" == CV ]]; then
    echo "Running CV"
elif [[ "$DEV_SUBSET" == test ]]; then
    echo "Running test"
fi

# TODO: generate config files for cross-validation of different models
ls "${CONFIGS_PAR_DIR}/${JOB_NAME}/${DEV_SUBSET}"

# All the jobs:
# joint_pooling, separate_pooling, joint_input_separate_pooling
# gLM2, ESM2, MINT

for representation in "joint_pooling" "separate_pooling" "joint_input_separate_pooling"; do
    for biolm in "gLM2" "ESM2" "MINT"; do
        if [[ "$representation" == joint_input_separate_pooling && "$biolm" == MINT ]]; then
            cp "${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/separate_pooling/${biolm}/${DEV_SUBSET}" \
               "${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/${representation}/${biolm}/${DEV_SUBSET}"
        else
            bash ./gLM/make_cv_config.sh "${CONFIGS_PAR_DIR}/${JOB_PAR_NAME}/${representation}/${biolm}/${DEV_SUBSET}" ""
        fi
    done
done

# TODO: generate the representations needed for the model

# TODO: collect the results: plot the metrics, pick the best epoch of the model

# TODO: generate config files for validation (full training + validation set)

# TODO: train fully until the chosen epoch and validate

# If dev. subset == test:

# TODO: generate config files for testing (change the run name, add ckpt_path)

# TODO: retrieve testing (point) metrics
