#!/usr/bin/env python3

# Preparation of the full training and validation configuration file

import yaml
import sys
import os
import re
from glob import glob

def remove_early_stopping(par_dir, dev_split="train_validate"):
    config = f"{par_dir}/{dev_split}/base.yaml"
    if not os.path.exists(config):
        print(f"{config} does not exist", sys.stderr)
        return 1
        
    with open(config) as f:
        content = yaml.safe_load(f)
        if 'trainer' in content and 'callbacks' in content['trainer']:
            content['trainer']['callbacks'] = [
                cb for cb in content['trainer']['callbacks']
                if not (isinstance(cb, dict) and cb.get('class_path') == 'pytorch_lightning.callbacks.EarlyStopping')
            ]
    return content

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

def write_config(config, par_dir, dev_split="train_validate"):
    config_path = f"{par_dir}/{dev_split}/base.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
par_dir = sys.argv[1]
num_folds = int(sys.argv[2])

config = remove_early_stopping(par_dir)
max_epoch = get_max_epochs(par_dir, num_folds=num_folds)
config = set_max_epochs(config, max_epoch)
write_config(config, par_dir)


