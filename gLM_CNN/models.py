from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
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

TOKENIZERS_PARALLELISM = False

class LogitsTensorGenerator(nn.Module):
    def __init__(self, model_path: str, config_path: str, fast: bool, 
        tensor_path: str, sep_chains=False, max_L=1024):
        super().__init__()
        self.fast = fast
        self.logits_type = 'fast' if(self.fast) else 'full'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # TODO: improve this determination
        if("gLM2" in model_path):
            self.LM = BioLM_gLM2(model_path)
            self.max_L = max_L + 2
        elif("esm2" in model_path):
            self.LM = ESM2(model_path)
            self.max_L = max_L + 3
        elif("mint" in model_path):
            self.LM = MINT(model_path, config_path, sep_chains)
            self.max_L = max_L + 4
        for param in self.LM.model.parameters():
            param.requires_grad = False

        self.tensor_path = tensor_path
        pathlib.Path(self.tensor_path).mkdir(parents=True, exist_ok=True)
    
    def pad(self, tensor):
        L, _, _, TT = tensor.shape
        pad_tot = self.max_L - L
        if pad_tot < 0:
            raise ValueError(f"L ({L}) is greater than L_max ({L_max})")
        pad_left = pad_tot // 2
        pad_right = pad_tot - pad_left        

        tensor = tensor.permute(1, 3, 0, 2)

        return F.pad(tensor, (pad_left, pad_right, pad_left, pad_right), mode='constant', value=0.0)

    def is_computed(self, id):
        tensor_path = pathlib.Path(f"{self.tensor_path}/{id}_{self.logits_type}Logits.pt")
        if(tensor_path.is_file() and tensor_path.stat().st_size != 0):
            return True

    def get_tensor(self, sequence: str, length1: int):
        input_ids, tokens, seqlen, chain_mask = self.LM.get_tokenized(sequence)
        masks = self.LM.get_masks(input_ids, seqlen, fast=self.fast)
        fx_h, fx = self.LM.get_logits(input_ids, chain_mask, fast=self.logits_type)

        L, T = fx.shape
        fx_expanded = fx.view(L, 1, 1, T).expand(L, 1, L, T)

        fx_h = self.LM.apply_masks(fx_h, masks)
        fx_expanded = self.LM.apply_masks(fx_expanded, masks)

        # Concatenating into shape: [L, 1, L, T+T]
        logits_tensor = torch.cat([fx_h, fx_expanded], dim=-1)
        
        # Padded: [1, T+T, max_L, max_L]
        logits_tensor = self.pad(logits_tensor)
        
        return logits_tensor

    def forward(self, x, x_idx, stage):
        tensors = None
        for i, s in enumerate(x['sequence']):
            if(self.is_computed(x['concat_id'][i])):
                tensor = torch.load(f"{self.tensor_path}/{x['concat_id'][i]}_{self.logits_type}Logits.pt")
            else:
                tensor = self.get_tensor(s, x['length1'][i])
                torch.save(tensor, f"{self.tensor_path}/{x['concat_id'][i]}_{self.logits_type}Logits.pt")
            tensors = torch.cat((tensors, tensor), dim=0) if(i) else tensor

        return tensors 

class LogitsTensorCNN(nn.Module):
    def __init__(self, tensor_dim: int, num_in_channels: int):
        super(LogitsTensorCNN, self).__init__()
        self.tensor_dim = tensor_dim
        self.num_in_channels = num_in_channels

        self.max_tensor_dim = 1028

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.contact_l = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_in_channels, 1, kernel_size=5, stride=1, padding=0),
            torch.nn.MaxPool2d(2, stride=2)
        )
        
        self.ppi_l = torch.nn.Sequential(
            torch.nn.Linear(512*512, 1),
            torch.nn.Sigmoid()
        )

    def get_input_pad(self, x):
        in_pad = self.max_tensor_dim - self.tensor_dim
        left_pad = in_pad // 2
        right_pad = in_pad - left_pad
        x = F.pad(x, pad=(left_pad, right_pad, left_pad, right_pad))
        return x

    def forward(self, x, x_idx, stage):
        x['logits_tensors'] = self.get_input_pad(x['logits_tensors'])
        x['logits_tensors'] = x['logits_tensors'].to(self.device)
        contact_l_out = self.contact_l(x['logits_tensors'])
        contact_predictor = torch.nn.Sigmoid()
        contact_pred = contact_predictor(contact_l_out)
        ppi_pred = self.ppi_l(torch.flatten(contact_l_out, start_dim=1))
        labels = torch.round(ppi_pred).int()
        return ppi_pred, labels, contact_pred     

    def compute_loss(self, x):
        x['predictions'] = x['predictions'].squeeze()
        binary_ppi_loss = torch.nn.functional.binary_cross_entropy(
            x['predictions'].to(self.device).float(),
            x['label'].to(self.device).float()
        )
        if(torch.isnan(x['urq']).any()):
            contact_loss = 0
        else:
            x['contact_pred'] = x['contact_pred'].squeeze()
            contact_loss = torch.nn.functional.binary_cross_entropy(
                x['contact_pred'].to(self.device).float(),
                x['urq'].to(self.device).float()
            )

        loss = binary_ppi_loss + contact_loss
        return loss

class PredictorPPI(LightningModule):
    def __init__(self, logits_model: nn.Module, model: nn.Module):
        super().__init__()
        self.logits_model = logits_model
        self.model = model
        self.save_hyperparameters()
        self.loss_accum = 0
        self.num_steps = 0
        self.configure_optimizers(self.model.parameters())

    def configure_optimizers(self, params):
        self.optimizer = torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.98), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        return {"optimizer": self.optimizer, "scheduler": self.scheduler, "monitor": "validate/loss"}
     
    def step(self, batch, batch_idx, split):
        batch['logits_tensors'] = self.logits_model(batch, batch_idx, split)
        batch['predictions'], batch['predicted_label'], batch['contact_pred'] = self.model(batch, batch_idx, stage=split)
   
        self.step_outputs[split] = batch
        loss = self.model.compute_loss(batch)
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
