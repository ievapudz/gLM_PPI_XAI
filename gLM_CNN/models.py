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
from gLM.models import CategoricalJacobian
import gc

TOKENIZERS_PARALLELISM = False

class LogitsTensorGenerator(nn.Module):
    def __init__(self, model_path: str, config_path: str, fast: bool, 
        tensor_path: str, sep_chains=False, max_L=1028):
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
    
    def pad(self, tensor):
        L, _, _, TT = tensor.shape
        pad_tot = self.max_L - L
        if pad_tot < 0:
            raise ValueError(f"L ({L}) is greater than max_L ({self.max_L})")
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
        
        # Reorganised and padded: [1, T+T, max_L, max_L]
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

class CategoricalJacobianURQGenerator(nn.Module):
    def __init__(self, model_path: str, config_path: str,
        cj_path: str, cj_type: str, distance: str, 
        sep_chains=False, n=3, max_L=512):
        
        super().__init__()

        self.cj_gen = CategoricalJacobian(
            model_path=model_path,
            config_path=config_path,
            fast=(cj_type == 'fast'),
            matrix_path=cj_path,
            distance=distance,
            sep_chains=sep_chains
        )

        self.cj_path = cj_path
        self.cj_type = cj_type
        self.n = n # Number of standard deviations for threshold
        self.max_L = max_L

    def pad(self, urq):
        m, n = urq.shape
        pad_hori = self.max_L - m
        pad_verti = self.max_L - n
        
        if pad_hori < 0 or pad_verti < 0:
            raise ValueError(f"m or n is greater than {self.max_L}")
        
        pad_top = pad_hori // 2
        pad_bot = pad_hori - pad_top
        pad_left = pad_verti // 2
        pad_right = pad_verti - pad_left
        
        return F.pad(urq, (pad_left, pad_right, pad_top, pad_bot), mode='constant', value=0.0)

    def get_urq(self, cj, length1: int, n: float):
        # Setting the threshold for the binarisation of the signals
        m = np.mean(cj)
        s = np.std(cj)
        threshold = m+n*s

        urq_cj_bin = None
        # Retrieving upper-right quadrant and skipping the tokens <+>
        if(isinstance(self.cj_gen.LM, BioLM_gLM2)):
            urq_cj = cj[1:length1+1, length1+2:]
            urq_cj_bin = np.where(urq_cj > threshold, 1, 0)
        elif(isinstance(self.cj_gen.LM, MINT)):
            urq_cj = cj[1:length1+1, length1+3:-1]
            urq_cj_bin = np.where(urq_cj > threshold, 1, 0)
        # TODO: implement ESM2 case

        urq_cj = torch.from_numpy(urq_cj)
        urq_cj = self.pad(urq_cj)
        urq_cj = torch.unsqueeze(urq_cj, dim=0)
 
        return urq_cj

    def forward(self, x, x_idx, stage):
        urqs = None
        for i, s in enumerate(x['sequence']):
            if(self.cj_gen.is_computed(x['concat_id'][i])):
                matrix = np.load(f"{self.cj_path}/{x['concat_id'][i]}_{self.cj_type}CJ.npy")
            else:
                matrix = self.cj_gen.get_matrix(s, x['length1'][i])
                np.save(f"{self.cj_path}/{x['concat_id'][i]}_{self.cj_type}CJ.npy", matrix)
            urq = self.get_urq(matrix, x['length1'][i], n=self.n)
            urqs = torch.cat((urqs, urq), dim=0) if(i) else urq

        return urqs

class CategoricalJacobianGenerator(nn.Module):
    def __init__(self, model_path: str, config_path: str,
        cj_path: str, cj_type: str, distance: str,
        sep_chains=False, n=3, max_L=1024):

        super().__init__()

        self.cj_gen = CategoricalJacobian(
            model_path=model_path,
            config_path=config_path,
            fast=(cj_type == 'fast'),
            matrix_path=cj_path,
            distance=distance,
            sep_chains=sep_chains
        )

        self.cj_path = cj_path
        self.cj_type = cj_type
        self.n = n # Number of standard deviations for threshold
        self.max_L = max_L

    def pad(self, matrix):
        n, _ = matrix.shape
        matrix = torch.from_numpy(matrix)
        pad = self.max_L - n

        if pad < 0:
            raise ValueError(f"n is greater than {self.max_L}")

        pad_1 = pad // 2
        pad_2 = pad - pad_1

        return F.pad(matrix, (pad_1, pad_2, pad_1, pad_2), mode='constant', value=0.0)

    def forward(self, x, x_idx, stage):
        matrices = None
        for i, s in enumerate(x['sequence']):
            if(self.cj_gen.is_computed(x['concat_id'][i])):
                matrix = np.load(f"{self.cj_path}/{x['concat_id'][i]}_{self.cj_type}CJ.npy")
            else:
                matrix = self.cj_gen.get_matrix(s, x['length1'][i])
                np.save(f"{self.cj_path}/{x['concat_id'][i]}_{self.cj_type}CJ.npy", matrix)
            matrix = self.pad(matrix)
            matrix = torch.unsqueeze(matrix, dim=0)
            matrices = torch.cat((matrices, matrix), dim=0) if(i) else matrix

        return matrices


class LogitsTensorCNN(nn.Module):
    def __init__(self, tensor_dim: int, num_in_channels: int):
        super(LogitsTensorCNN, self).__init__()
        self.tensor_dim = tensor_dim
        self.num_in_channels = num_in_channels

        self.max_tensor_dim = 1028

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        out_channels = [256, 256, 256, 1]

        last_dim = self.max_tensor_dim
        for i, k in enumerate(kernel_sizes):
            last_dim = int((last_dim - k)/strides[i] + 1)

        self.contact_l = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(1),
            torch.nn.Conv2d(self.num_in_channels, out_channels[0], kernel_size=kernel_sizes[0], padding=0),
            nn.LeakyReLU(0.01),
            torch.nn.Conv2d(out_channels[0], out_channels[1], kernel_size=kernel_sizes[1], padding=0),
            nn.LeakyReLU(0.01),
            torch.nn.Conv2d(out_channels[1], out_channels[2], kernel_size=kernel_sizes[2], padding=0),
        )

        self.ppi_l = torch.nn.Sequential(
            nn.LeakyReLU(0.01),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(last_dim*last_dim*out_channels[-2], out_channels[-1]),
        )

    def get_input_pad(self, x):
        # TODO: to do full padding within the generator class
        #       It was done here additionally because the logits
        #       were generated with too little padding for the chosen
        #       architecture.
        in_pad = self.max_tensor_dim - self.tensor_dim
        left_pad = in_pad // 2
        right_pad = in_pad - left_pad
        x = F.pad(x, pad=(left_pad, right_pad, left_pad, right_pad))
        return x

    def forward(self, x, x_idx, stage):
        x['input'] = self.get_input_pad(x['input'])
        x['input'] = x['input'].squeeze().to(self.device)
        contact_l_out = self.contact_l(x['input'])
        contact_predictor = torch.nn.Sigmoid()
        contact_pred = contact_predictor(contact_l_out).squeeze()
        contact_pred = contact_predictor(contact_l_out)
        ppi_pred = self.ppi_l(torch.flatten(contact_l_out, start_dim=1))
        labels = torch.round(ppi_pred).int()
        return ppi_pred, labels, contact_pred     

    def compute_loss(self, x):
        loss = torch.nn.BCEWithLogitsLoss()

        x['predictions'] = x['predictions'].squeeze()
        binary_ppi_loss = loss(
            x['predictions'].to(self.device).float(),
            x['label'].to(self.device).float()
        )
        if(torch.isnan(x['urq']).any()):
            contact_loss = 0
        else:
            x['contact_pred'] = x['contact_pred'].squeeze()
            contact_loss = loss(
                x['contact_pred'].to(self.device).float(),
                x['urq'].to(self.device).float()
            )

        loss = binary_ppi_loss + contact_loss
        return loss

class CategoricalJacobianURQCNN(nn.Module):
    def __init__(self, matrix_dim: int):
        super(CategoricalJacobianURQCNN, self).__init__()
        self.matrix_dim = matrix_dim 

        self.max_matrix_dim = 513

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layers = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(1),
            torch.nn.Conv2d(1, 2, kernel_size=9, stride=4, padding=0),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Conv2d(2, 2, kernel_size=5, stride=2, padding=0),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(62*62*2, 1),
            torch.nn.Dropout(0.8),
        )
        self.layers_2 = torch.nn.Sequential(
            torch.nn.Sigmoid()
        )

        # Initialisation of the weights
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
                    
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x, x_idx, stage):
        x['input'] = torch.unsqueeze(x['input'], dim=1).to(self.device)
        
        intermed = self.layers(x['input'])
        ppi_pred = self.layers_2(intermed)
        labels = torch.round(ppi_pred).int()
        labels = torch.squeeze(labels)
        return ppi_pred, labels, None

    def compute_loss(self, x):
        x['predictions'] = x['predictions'].squeeze()

        binary_ppi_loss = torch.nn.functional.binary_cross_entropy(
            x['predictions'].to(self.device).float(),
            x['label'].to(self.device).float()
        )

        loss = binary_ppi_loss
        return loss

class CategoricalJacobianCNN(nn.Module):
    def __init__(self, matrix_dim: int, loss: str, 
            kernel_sizes: list, strides: list, out_channels: list, 
            num_linear_layers: int, dropout: float
        ):
        super(CategoricalJacobianCNN, self).__init__()
        self.matrix_dim = matrix_dim
        self.loss = loss

        self.max_matrix_dim = 1026
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        last_dim = self.matrix_dim
        for i, k in enumerate(kernel_sizes):
            last_dim = int((last_dim - k)/strides[i] + 1)

        self.layers = torch.nn.Sequential()

        # Final activation function
        if(self.loss == "BCE"):
            self.layers_2 = torch.nn.Sequential(
                torch.nn.Sigmoid()
            )
            out_channels.append(1)
        elif(self.loss == "CE"):
            self.layers_2 = torch.nn.Sequential(
                torch.nn.Softmax(dim=1)
            )
            out_channels.append(2)

        # Adding CNN layers
        for i, ks in enumerate(kernel_sizes):
            if(i):
                self.layers.append(
                    torch.nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=ks, stride=strides[i], padding=0)
                )
            else:
                # At i == 0
                self.layers.append(
                    torch.nn.Conv2d(1, out_channels[i], kernel_size=ks, stride=strides[i], padding=0)
                )
            self.layers.append(nn.LeakyReLU(0.01))
        
        self.layers.append(torch.nn.Flatten(start_dim=1))

        # Adding linear layers
        for i in range(num_linear_layers):
            if(i == num_linear_layers-1):
                self.layers.append(
                    torch.nn.Linear(
                        int(last_dim*last_dim*out_channels[-2]/2**(i)),
                        out_channels[-1]
                    )
                )
            else:
                self.layers.append(
                    torch.nn.Linear(
                        int(last_dim*last_dim*out_channels[-2]/2**(i)),
                        int(last_dim*last_dim*out_channels[-2]/2**(i+1))
                    )
                )

        # Dropout
        self.layers.append(torch.nn.Dropout(dropout))
                 
        # Initialisation of the weights
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, x_idx, stage):
        x['input'] = torch.unsqueeze(x['input'], dim=1).to(self.device)

        if(self.loss == "BCE"):
            ppi_pred = self.layers(x['input'])
            labels = torch.round(self.layers_2(ppi_pred)).int()
        elif(self.loss == "CE"):
            ppi_pred = self.layers(x['input'])
            labels = torch.round(self.layers_2(ppi_pred)[:,1])
                
        labels = torch.squeeze(labels)
        return ppi_pred, labels, None

    def compute_loss(self, x):
        x['predictions'] = x['predictions'].squeeze()

        if(self.loss == "BCE"):
            loss = torch.nn.BCEWithLogitsLoss()
            binary_ppi_loss = loss(
                x['predictions'].to(self.device).float(),
                x['label'].to(self.device).float()
            )
        elif(self.loss == "CE"):
            loss = torch.nn.CrossEntropyLoss()
            binary_ppi_loss = loss(
                x['predictions'].to(self.device),
                x['label'].to(self.device)
            )
        
            x['predictions'] = x['predictions'][:,1]
        
        loss = binary_ppi_loss
        return loss

class PredictorPPI(LightningModule):
    def __init__(self, logits_model: nn.Module, model: nn.Module):
        super().__init__()
        self.logits_model = logits_model
        self.model = model
        self.save_hyperparameters()
        self.train_loss_accum = 0
        self.train_num_steps = 0
        self.val_loss_accum = 0
        self.val_num_steps = 0

        self.epoch_outputs = {
            'train': {
                'concat_id': [], 'label': [], 'predicted_label': [], 'predictions': []
            },
            'validate': {
                'concat_id': [], 'label': [], 'predicted_label': [], 'predictions': []
            }, 
            'test': {
                'concat_id': [], 'label': [], 'predicted_label': [], 'predictions': []
            }
        }

    def step(self, batch, batch_idx, split):
        batch['input'] = self.logits_model(batch, batch_idx, split)
        batch['predictions'], batch['predicted_label'], batch['contact_pred'] = self.model(batch, batch_idx, stage=split)
   
        loss = self.model.compute_loss(batch)
        self.log(f'{split}/loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        for key in self.epoch_outputs[split]:
            if key in batch:
                self.epoch_outputs[split][key].extend(batch[key])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        self.train_loss_accum += loss.detach().cpu().item()
        self.train_num_steps += 1
        return loss

    def on_train_epoch_end(self):
        print(f"epoch train loss: {self.train_loss_accum/self.train_num_steps:.3f}")
        self.train_loss_accum = 0
        self.train_num_steps = 0

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'validate')
        self.val_loss_accum += loss.detach().cpu().item()
        self.val_num_steps += 1
        return loss

    def on_validation_epoch_end(self):
        print(f"epoch val loss: {self.val_loss_accum/self.val_num_steps:.3f}")
        self.val_loss_accum = 0
        self.val_num_steps = 0

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        return loss
