from pytorch_lightning import LightningModule
from torch import nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import numpy as np
import pandas as pd
import math
from sklearn.metrics import zero_one_loss
from pytorch_lightning import Trainer
import pathlib
from scipy.ndimage import gaussian_filter1d
from torch.distributions import Categorical
import scipy.ndimage as ndimage
import os
from gLM.LMs import gLM2 as BioLM_gLM2
from gLM.LMs import ESM2
from gLM.LMs import MINT

TOKENIZERS_PARALLELISM = True

class LogitsTensorGenerator(nn.Module):
    def __init__(self, model_path: str, config_path: str, fast: bool, 
        tensor_path: str, sep_chains=False):
        super().__init__()
        self.fast = fast
        self.logits_type = 'fast' if(self.fast) else 'full'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # TODO: improve this determination
        if("gLM2" in model_path):
            self.LM = BioLM_gLM2(model_path)
        elif("esm2" in model_path):
            self.LM = ESM2(model_path)
        elif("mint" in model_path):
            self.LM = MINT(model_path, config_path, sep_chains)

        for param in self.LM.model.parameters():
            param.requires_grad = False

        self.tensor_path = tensor_path
        pathlib.Path(self.tensor_path).mkdir(parents=True, exist_ok=True)
    
    def is_computed(self, id):
        tensor_path = pathlib.Path(f"{self.tensor_path}/{id}_{self.logits_type}Logits.pt")
        if(tensor_path.is_file() and tensor_path.stat().st_size != 0):
            return True

    def get_tensor(self, sequence: str, length1: int):
        input_ids, tokens, seqlen, chain_mask = self.LM.get_tokenized(sequence)
        masks = self.LM.get_masks(input_ids, seqlen)
        fx_h, fx = self.LM.get_logits(input_ids, chain_mask, fast=self.logits_type)

        L, T = fx.shape
        fx_expanded = fx.view(L, 1, 1, T).expand(L, 1, L, T)

        # Concatenating into shape: [L, 1, L, T+T]
        logits_tensor = torch.cat([fx_h, fx_expanded], dim=-1)

        return logits_tensor

    def forward(self, x, x_idx, stage):
        for i, s in enumerate(x['sequence']):
            if(self.is_computed(x['concat_id'][i])):
                tensor = torch.load(f"{self.tensor_path}/{x['concat_id'][i]}_{self.logits_type}Logits.pt")
            else:
                tensor = self.get_tensor(s, x['length1'][i])
                print(tensor.shape)
                torch.save(tensor, f"{self.tensor_path}/{x['concat_id'][i]}_{self.logits_type}Logits.pt")

        return tensor 

class PredictorPPI(LightningModule):
    def __init__(self, logits_model: nn.Module):
        super().__init__()
        self.logits_model = logits_model
        #self.model = model
        self.save_hyperparameters()
        self.loss_accum = 0
        self.num_steps = 0
        #self.configure_optimizers(self.model.parameters())
     
    def step(self, batch, batch_idx, split):
        batch['logits_tensors'] = self.logits_model(batch, batch_idx, split)
        print(batch['logits_tensors'])
        batch['predictions'], batch['predicted_label'] = self.model(batch, batch_idx, stage=split)
   
        self.step_outputs[split] = batch
        loss = 0
        self.log(f'{split}/loss', loss.detach().cpu().item(), on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        self.loss_accum += loss.detach().cpu().item()
        self.num_steps += 1
        return loss

    def on_train_epoch_end(self):
        self.scheduler.step(self.loss_accum / self.num_steps + 1)
        self.loss_accum = 0
        self.num_steps = 0
        self.step_outputs["train"].clear()
        self.step_outputs["validate"].clear()

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'validate')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        return loss
