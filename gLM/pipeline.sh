#!/bin/bash

# Automated pipeline for cross-validation and testing of PPI models (all LMs)

# Requirements: prepare the base.yaml file

# TODO: take job name as input: the name of the set together with dev. subset
#       dev. subset: CV or test

JOB_NAME=$1
DEV_SUBSET=$2

if [[ "$DEV_SUBSET" == CV ]]; then
    echo "Running CV"
elif [[ "$DEV_SUBSET" == test ]]; then
    echo "Running test"
fi

# TODO: generate config files for cross-validation of different models

# TODO: generate the representations needed for the model

# TODO: collect the results: plot the metrics, pick the best epoch of the model

# TODO: generate config files for validation (full training + validation set)

# TODO: train fully until the chosen epoch and validate

# If dev. subset == test:

# TODO: generate config files for testing (change the run name, add ckpt_path)

# TODO: retrieve testing (point) metrics
