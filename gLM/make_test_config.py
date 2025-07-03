#!/usr/bin/env python3

# Preparation of the full training and validation configuration file

import yaml
import sys
import os
import re
from glob import glob
from optparse import OptionParser
import pandas as pd
import numpy as np

parser = OptionParser()

parser.add_option("--config", "-c", dest="config_dir_path",
	default=None, help="path to the config. files' directory.")

parser.add_option("--output", "-o", dest="output_dir_path",
	default=None, help="path to the output files' directory.")

parser.add_option("--job-name", "-j", dest="job_name",
	default=None, help="name of the job.")

parser.add_option("--out-job-name", dest="out_job_name",
	default=None, help="name of the output job.")

parser.add_option("--representation", "-r", dest="representation",
	default=None, help="representation of interest.")

parser.add_option("--biolm", "-b", dest="biolm",
	default=None, help="biolm of interest.")

parser.add_option("--hyperparam", dest="hyperparam",
	default=None, help="hyperparameter to consider to find the best model.")

(options, args) = parser.parse_args()

def read_base_config(config_dir):
    config = f"{config_dir}/base.yaml"
    if not os.path.exists(config):
        print(f"{config} does not exist", sys.stderr)
        return 1
        
    with open(config) as f:
        content = yaml.safe_load(f)
        
    return content

def remove_early_stopping(config):
    if 'trainer' in config and 'callbacks' in config['trainer']:
        config['trainer']['callbacks'] = [
            cb for cb in config['trainer']['callbacks']
            if not (isinstance(cb, dict) and cb.get('class_path') == 'pytorch_lightning.callbacks.EarlyStopping')
        ]
    return config

def get_best_hyperparam(param_config, hyperparam):
    if(hyperparam == "batch_size"):
        return param_config['data']['init_args'][hyperparam]
    elif(hyperparam == "optimizer"):
        return param_config[hyperparam]['class_path']
    else:
        print(f"Hyperparam {hyperparam} cannot be processed.")
        return None

def get_best_model_path(out_dir, representation, biolm, dev_split="train_validate"):
    pattern = f"{out_dir}/{representation}/{biolm}/{dev_split}/checkpoints/model-epoch=*.ckpt"
    best_model_paths = glob(pattern)

    if not best_model_paths:
        raise FileNotFoundError(f"No metric files found matching pattern: {pattern}")

    return best_model_paths[0]

def set_best_hyperparam(config, hyperparam, best_hyperparam_value):
    if(hyperparam == "batch_size"):
        config['data']['init_args'][hyperparam] = best_hyperparam_value
    elif(hyperparam == "optimizer"):
        config[hyperparam]['class_path'] = best_hyperparam_value
    return config

def set_ckpt_path(config, path):
    config['ckpt_path'] = path
    return config

def write_config(config, config_dir):
    config_path = f"{config_dir}/base.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
param_config_dir = f"{options.config_dir_path}/{options.job_name}/{options.representation}/{options.biolm}/train_validate/"

test_config_dir = f"{options.config_dir_path}/{options.out_job_name}/{options.representation}/{options.biolm}/test/"
logs_dir = f"{options.output_dir_path}/{options.job_name}/"

hyperparams = options.hyperparam.split(' ')

config = read_base_config(test_config_dir)
config = remove_early_stopping(config)
best_model_path = get_best_model_path(logs_dir,
    options.representation, options.biolm, dev_split="train_validate", 
)
config = set_ckpt_path(config, best_model_path)

param_config = read_base_config(param_config_dir)

for hyperparam in hyperparams:
    hyperparam_value = get_best_hyperparam(param_config, hyperparam)
    config = set_best_hyperparam(config, hyperparam, hyperparam_value)

os.makedirs(test_config_dir, exist_ok=True)  
write_config(config, test_config_dir)


