#!/usr/bin/env python3

# Preparation of the CJ-based predictor's testing configuration file

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

parser.add_option("--representation", "-r", dest="representation",
	default=None, help="representation of interest.")

parser.add_option("--biolm", "-b", dest="biolm",
	default=None, help="biolm of interest.")

parser.add_option("--hyperparam", dest="hyperparam",
	default=None, help="hyperparameter to consider to find the best model.")

(options, args) = parser.parse_args()

def read_base_config(par_dir, dev_split="test"):
    config = f"{par_dir}/{dev_split}/base.yaml"
    
    if not os.path.exists(config):
        print(f"{config} does not exist", sys.stderr)
        return 1
        
    with open(config) as f:
        content = yaml.safe_load(f)
        
    return content

def get_best_hyperparams(
    base_path: str,
    biolm: str,
    metric: str = "mcc",
    param_name: str = "n",
    plot: bool = True
):
    # Search for all relevant metric files
    pattern = f"{base_path}/{param_name}_*/{biolm}/validate/metrics.csv"
    csv_files = glob(pattern)

    if not csv_files:
        raise FileNotFoundError(f"No files matched the pattern: {pattern}")

    records = []

    for file in csv_files:
        match = re.search(fr"{param_name}_([-+]?\d*\.?\d+)", file)
        if not match:
            continue
        n_value = float(match.group(1))
        df = pd.read_csv(file)

        if df.empty or metric not in df.columns:
            continue

        mean_metric = df[metric].mean()
        stderr_metric = df[metric].std(ddof=1) / (len(df) ** 0.5)

        records.append({
            param_name: n_value,
            "biolm": biolm,
            metric: mean_metric,
            "stderr": stderr_metric
        })

    if not records:
        raise ValueError("No valid metric data found.")

    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(by=param_name)

    # Select best n — choose largest n with highest metric value
    max_metric = result_df[metric].max()
    best_candidates = result_df[result_df[metric] == max_metric]
    best_n = best_candidates[param_name].max()

    # TODO: adjust to save the plot
    if plot:
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=result_df,
            x=param_name,
            y=metric,
            marker="o",
            label=biolm
        )
        plt.axvline(best_n, color='red', linestyle='--', label=f'Best n = {best_n}')
        plt.title(f"{metric.upper()} vs. {param_name} for {biolm}")
        plt.xlabel(param_name)
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return best_n, result_df

def set_best_hyperparam(config, hyperparam, best_hyperparam_value):
    if(hyperparam == "n"):
        config['model']['init_args']['model']['init_args'][hyperparam] = float(best_hyperparam_value)
    return config

def write_config(config, par_dir, dev_split="test"):
    config_path = f"{par_dir}/{dev_split}/base.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
par_dir = f"{options.config_dir_path}/{options.job_name}/{options.representation}/{options.biolm}"
out_dir = f"{options.output_dir_path}/{options.job_name}/{options.representation}/"
hyperparam = options.hyperparam

config = read_base_config(par_dir)

best_hyperparam, df = get_best_hyperparams(
    base_path=out_dir,
    biolm=options.biolm,
    metric="mcc",
    param_name=hyperparam,
    plot=False
)

config = set_best_hyperparam(config, hyperparam, 
    best_hyperparam_value=best_hyperparam
)

write_config(config, par_dir)


