#!/usr/bin/env python

import sys
import os

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import seed_everything
import torch
import warnings
from jsonargparse import ArgumentParser, Namespace
import mlflow

warnings.filterwarnings('ignore')

os.environ["WANDB_SILENT"] = "true"

def main():
	"""
	Run with python main.py test -c configs/config.yaml
	"""
	torch.set_float32_matmul_precision('medium')
	cli = LightningCLI(save_config_kwargs={"overwrite": True})

if __name__ == '__main__':
	main()
