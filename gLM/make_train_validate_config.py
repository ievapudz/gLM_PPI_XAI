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

parser.add_option("--folds", "-f", dest="num_folds",
	default=5, help="number of CV folds to consider.")

parser.add_option("--job-name", "-j", dest="job_name",
	default=None, help="name of the job.")

parser.add_option("--representation", "-r", dest="representation",
	default=None, help="representation of interest.")

parser.add_option("--biolm", "-b", dest="biolm",
	default=None, help="biolm of interest.")

parser.add_option("--hyperparam", dest="hyperparam",
	default=None, help="hyperparameter to consider to find the best model.")

(options, args) = parser.parse_args()

def read_base_config(par_dir, dev_split="train_validate"):
    config = f"{par_dir}/{dev_split}/base.yaml"
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

def get_max_epochs(par_dir, dev_split="CV", num_folds=5):
    best_epochs = []
    for k in range(num_folds):
        fold_config = f"{par_dir}/{dev_split}/{k}.yaml"
        if not os.path.exists(fold_config):
            print(f"{fold_config} does not exist", sys.stderr)
            return 1

        with open(fold_config) as f:
            content = yaml.safe_load(f)
            checkpoints_path = [cb['init_args']['dirpath'] for cb in content['trainer']['callbacks'] 
                if (isinstance(cb, dict) and cb.get('class_path') == 'pytorch_lightning.callbacks.ModelCheckpoint')
            ][0]
            ckpt_path = glob(f"{checkpoints_path}/model-epoch=*.ckpt")
            match = re.search(r"model-epoch=(\d+)\.ckpt", os.path.basename(ckpt_path[0]))
            best_epoch = int(match.group(1))
            best_epochs.append(best_epoch)
    
    avg_epoch = sum(best_epochs)/num_folds
    max_epoch = int(avg_epoch) + bool(avg_epoch%1) 
    return max_epoch

def set_max_epochs(config, max_epoch):
    config['trainer']['max_epochs'] = max_epoch
    return config

def get_best_hyperparams(out_dir, hyperparams, representation, biolm, 
                         dev_split="CV", target_metric="mcc"):
    # Find all candidate folders
    pattern = out_dir + "/" + "/".join([f"{h}_*" for h in hyperparams]) + f"/{dev_split}/metrics.csv"
    metric_paths = glob(pattern)

    if not metric_paths:
        raise FileNotFoundError(f"No metric files found matching pattern: {pattern}")

    results = []

    for path in metric_paths:
        match = [re.search(fr"{h}_([^/\\]+)", path) for h in hyperparams]
        if any(m is None for m in match):
            continue
        values = [m.group(1) for m in match]
        
        df = pd.read_csv(path)
        filtered = df[(df["representation"] == representation) & (df["biolm"] == biolm)]
        if filtered.empty:
            continue

        mean = filtered[target_metric].mean()
        stderr = filtered[target_metric].std(ddof=1) / np.sqrt(filtered.shape[0])
        proc_mean = mean - stderr
        
        results.append((tuple(values), proc_mean))

    if not results:
        raise ValueError(f"No results found for representation={representation}, biolm={biolm}")

    # Sort by processed mean descending
    results.sort(key=lambda x: -x[1])

    best_values = results[0][0]
    return dict(zip(hyperparams, best_values))

def set_best_hyperparam(config, hyperparam, best_hyperparam_value):
    if(hyperparam == "batch_size"):
        config['data']['init_args'][hyperparam] = best_hyperparam_value
    elif(hyperparam == "optimizer"):
        config[hyperparam]['class_path'] = f"torch.optim.{best_hyperparam_value}"
    return config

def write_config(config, par_dir, dev_split="train_validate"):
    config_path = f"{par_dir}/{dev_split}/base.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
par_dir = f"{options.config_dir_path}/{options.job_name}/{options.representation}/{options.biolm}/"
out_dir = f"{options.output_dir_path}/{options.job_name}/"
num_folds = int(options.num_folds)
hyperparams = options.hyperparam.split(" ")

config = read_base_config(par_dir)

if(options.hyperparam == "epoch"):
    config = remove_early_stopping(config)
    max_epoch = get_max_epochs(par_dir, num_folds=num_folds)
    config = set_max_epochs(config, max_epoch)
else:
    best_hyperparam = get_best_hyperparams(out_dir, hyperparams,
        options.representation, options.biolm, dev_split="CV"
    )
    for hparam in hyperparams:
        config = set_best_hyperparam(config, hparam, 
            best_hyperparam_value=best_hyperparam[hparam]
        )
    
write_config(config, par_dir)


