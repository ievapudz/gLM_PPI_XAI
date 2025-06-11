#!/usr/bin/env python3
"""
Simple Grid Search Configuration Generator
Creates config files from CSV parameters with array support.
"""

import yaml
import csv
import sys
import os
import re

def parse_value(value):
    """Parse a value that could be a number, array, or string."""
    if isinstance(value, str):
        value = value.strip()
        # Check if it's an array (starts with [ and ends with ])
        if value.startswith('[') and value.endswith(']'):
            # Parse as array
            array_str = value[1:-1]  # Remove brackets
            return [float(x.strip()) if '.' in x.strip() else int(x.strip()) 
                   for x in array_str.split(' ') if x.strip()]
        else:
            # Try to parse as number
            try:
                return float(value) if '.' in value else int(value)
            except ValueError:
                return value
    return value


def expand_to_array(value, length):
    """Expand a single value to an array of given length, or return array as-is."""
    if isinstance(value, list):
        return value
    else:
        return [value] * length


def generate_config(base_config, params, prefix, model_log_dir_prefix="logs/CategoricalJacobianCNN/"):
    """Generate a new config with the given parameters."""
    config = yaml.safe_load(open(base_config))
    
    # Parse parameters
    exp_id = params['id']
    num_layers = int(params['num_cnn_layers'])
    
    # Parse potentially array values
    kernel_size = parse_value(params['kernel_size'])
    strides = parse_value(params['strides'])
    out_channels = parse_value(params['num_out_channels'])
    dropout = parse_value(params['dropout'])
    
    # Expand to arrays
    kernel_sizes = expand_to_array(kernel_size, num_layers)
    strides_array = expand_to_array(strides, num_layers)
    out_channels_array = expand_to_array(out_channels, num_layers)
    
    # Update config
    model_args = config['model']['init_args']['model']['init_args']
    model_args['kernel_sizes'] = kernel_sizes
    model_args['strides'] = strides_array
    model_args['out_channels'] = out_channels_array
    model_args['num_linear_layers'] = int(params['num_linear_layers'])
    model_args['dropout'] = dropout
    
    config['optimizer']['init_args']['lr'] = float(params['lr'])
    
    # Update experiment names
    config['trainer']['logger']['init_args']['name'] += f"_{exp_id}"
  
    # Update path for the logging directory 
    orig_path = config['trainer']['logger']['init_args']['save_dir']
    config['trainer']['logger']['init_args']['save_dir'] = re.sub(
        rf'{model_log_dir_prefix}{prefix}_00/',
        rf'{model_log_dir_prefix}{prefix}_{exp_id}/',
        orig_path
    )
 
    # Update paths for the callbacks' files
    for i, callback in enumerate(config['trainer']['callbacks']):
        if(callback['class_path'] == 'pytorch_lightning.callbacks.ModelCheckpoint'):
            orig_path = callback['init_args']['dirpath']
            config['trainer']['callbacks'][i]['init_args']['dirpath'] = re.sub(
                rf'{model_log_dir_prefix}{prefix}_00/',
                rf'{model_log_dir_prefix}{prefix}_{exp_id}/',
                orig_path
            )
        elif(callback['class_path'] == 'gLM.callbacks.OutputLoggingCallback'):
            orig_file = callback['init_args']['metric_log_file']
            config['trainer']['callbacks'][i]['init_args']['metric_log_file'] = re.sub(
                rf'{model_log_dir_prefix}{prefix}_00/',
                rf'{model_log_dir_prefix}{prefix}_{exp_id}/',
                orig_file
            )
    return config


def main():
    if len(sys.argv) != 5:
        print("Usage: python configs/make_grid_search_base.py <base_config.yaml> <grid_params.csv> <output_dir>")
        sys.exit(1)
    
    base_config_file = sys.argv[1]
    csv_file = sys.argv[2]
    output_dir = sys.argv[3]
    prefix = sys.argv[4]
    
    # Read CSV and generate configs
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = generate_config(base_config_file, row, prefix)
            
            # Save config
            os.makedirs(f"{output_dir}/{prefix}_{row['id']}", exist_ok=True)
            output_file = f"{output_dir}/{prefix}_{row['id']}/base.yaml"
            with open(output_file, 'w') as out:
                yaml.dump(config, out, default_flow_style=False, sort_keys=False)
            
            print(f"Generated: {output_file}")
    
    print(f"All configs generated in {output_dir}/{prefix}_*/")


if __name__ == "__main__":
    main()
