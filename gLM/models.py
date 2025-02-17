from pytorch_lightning import LightningModule
from torch import nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import pandas as pd
import math
from sklearn.metrics import zero_one_loss
from pytorch_lightning import Trainer
import mlflow

class CategoricalJacobian(nn.Module):
    def __init__(self, fast: bool):
        super().__init__()
        self.fast = fast
        self.nuc_tokens = tuple(range(29, 33)) # 4 nucleotides a,t,c,g
        self.aa_tokens = tuple(range(4, 24)) # 20 amino acids

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # TODO: consider setting it in a customizable way
        model_path = "./gLM2_650M"
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.MASK_TOKEN_ID = self.tokenizer.mask_token_id

        for param in self.model.parameters():
            param.requires_grad = False

    def jac_to_contact(self, jac, symm=True, center=True, diag="remove", apc=True):
        X = jac.copy()
        Lx, Ax, Ly, Ay = X.shape

        if center:
            for i in range(4):
                if X.shape[i] > 1:
                    X -= X.mean(i, keepdims=True)

        contacts = np.sqrt(np.square(X).sum((1,3)))

        if symm and (Ax != 20 or Ay != 20):
            contacts = (contacts + contacts.T)/2

        if diag == "remove":
            np.fill_diagonal(contacts,0)

        if diag == "normalize":
            contacts_diag = np.diag(contacts)
            contacts = contacts/np.sqrt(contacts_diag[:,None]* contacts_diag[None,:])

        if apc:
            ap = contacts.sum(0,keepdims=True)*contacts.sum(1, keepdims=True)/contacts.sum()
            contacts = contacts - ap

        if diag == "remove":
            np.fill_diagonal(contacts,0)

        return contacts

    def get_categorical_jacobian(self, sequence: str):
        all_tokens = self.nuc_tokens + self.aa_tokens
        num_tokens = len(all_tokens)

        input_ids = torch.tensor(self.tokenizer.encode(sequence), dtype=torch.int)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        seqlen = input_ids.shape[0]
        # [seqlen, 1, seqlen, 1].
        is_nuc_pos = torch.isin(input_ids, torch.tensor(self.nuc_tokens)).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
        # [1, num_tokens, 1, num_tokens].
        is_nuc_token = torch.isin(torch.tensor(all_tokens), torch.tensor(self.nuc_tokens)).view(1, -1, 1, 1).repeat(1, 1, 1, num_tokens)
        # [seqlen, 1, seqlen, 1].
        is_aa_pos = torch.isin(input_ids, torch.tensor(self.aa_tokens)).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
        # [1, num_tokens, 1, num_tokens].
        is_aa_token = torch.isin(torch.tensor(all_tokens), torch.tensor(self.aa_tokens)).view(1, -1, 1, 1).repeat(1, 1, 1, num_tokens)

        input_ids = input_ids.unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
            f = lambda x: self.model(x)[0][..., all_tokens].cpu().float()

            x = torch.clone(input_ids).to(self.device)
            ln = x.shape[1]

            fx = f(x)[0]
            if self.fast:
                fx_h = torch.zeros((ln, 1 , ln, num_tokens), dtype=torch.float32)
            else:
                fx_h = torch.zeros((ln,num_tokens,ln,num_tokens),dtype=torch.float32)
                x = torch.tile(x,[num_tokens,1])

            for n in range(ln): # for each position
                x_h = torch.clone(x)
                if self.fast:
                    x_h[:, n] = self.MASK_TOKEN_ID
                else:
                    x_h[:, n] = torch.tensor(all_tokens)
                fx_h[n] = f(x_h)

            jac = fx_h-fx
            valid_nuc = is_nuc_pos & is_nuc_token
            valid_aa = is_aa_pos & is_aa_token
            # Zero out other modality
            jac = torch.where(valid_nuc | valid_aa, jac, 0.0)
            contact = self.jac_to_contact(jac.numpy())
        return jac, contact, tokens 
    
    def contact_to_dataframe(self, con):
        sequence_length = con.shape[0]
        idx = [str(i) for i in np.arange(1, sequence_length+1)]
        df = pd.DataFrame(con, index=idx, columns=idx)
        df = df.stack().reset_index()
        df.columns = ['i', 'j', 'value']
        return df
    
    def sigmoid(self, x):
        return 1/(1+math.exp(-x))
    
    def detect_ppi(self, array, len1, sigmoid):
        # Computing contact probability
        array = sigmoid(array)
        # Detecting the PPI signal in upper right quadrant of matrix
        return np.max(array[:len1, len1:])

    def forward(self, x):
        sigmoid_v = np.vectorize(self.sigmoid)
        ppi_preds = []
        
        for i, s in enumerate(x['sequence']):
            J, contact, tokens = self.get_categorical_jacobian(s)
            df = self.contact_to_dataframe(contact)

            # TODO: perhaps this chunk of code could be optimized?
            pivot_df = df.pivot(index='i', columns='j', values='value')

            sorted_cols = sorted([int(item) for item in pivot_df.columns], key=int)
            sorted_cols = [str(item) for item in sorted_cols]
            pivot_df = pivot_df[sorted_cols]

            # Sorting the rows
            pivot_df.index = pivot_df.index.astype(int)
            pivot_df = pivot_df.sort_index()

            # Convert the pivot table to a 2D numpy array
            array_2d = pivot_df.to_numpy()
            np.save(f"{x['concat_id'][i]}_CJ.npy", array_2d)

            # Detect the PPI signal in the CJ
            ppi_pred = self.detect_ppi(array_2d, x['length1'][i], sigmoid_v)
            ppi_preds.append(ppi_pred) 

        return torch.FloatTensor(ppi_preds)
    
    def compute_loss(self, x):
        predictions = self(x)
        pred_labels = np.round(predictions)
        return {'loss': zero_one_loss(x['label'], pred_labels)}

class gLM2(LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.save_hyperparameters()

    def get_log_outputs(self, x):
        return {'predictions': self.model(x)}
    
    def step(self, batch, batch_idx, split):
        batch['predictions'] = self.model(batch)
        self.step_outputs[split] = batch
        #losses = self.model.compute_loss(batch)
        return None

    def training_step(self, batch, batch_idx):
        self.step(batch, batch_idx, 'train')
        return None

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

