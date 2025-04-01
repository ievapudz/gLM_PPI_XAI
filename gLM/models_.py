from pytorch_lightning import LightningModule
from torch import nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import numpy as np
import pandas as pd
from sklearn.metrics import zero_one_loss
from pytorch_lightning import Trainer
import pathlib
import os
from gLM.LMs import gLM2
from gLM.LMs import ESM2

TOKENIZERS_PARALLELISM = True

class CategoricalJacobian(nn.Module):
    def __init__(self, model_path: str, fast: bool, matrix_path: str, distance: str):
        super().__init__()
        self.fast = fast
        self.cj_type = 'fast' if(self.fast) else 'full'

        if("gLM2" in model_path):
            self.LM = gLM2(model_path)
        elif("esm2" in model_path):
            self.LM = ESM2(model_path)

        for param in self.LM.model.parameters():
            param.requires_grad = False

        self.distance = distance if(distance) else "Euclidean"

        self.matrix_path = matrix_path
        pathlib.Path(self.matrix_path).mkdir(parents=True, exist_ok=True)

    def get_logits(self, input_ids):
        input_ids = input_ids.unsqueeze(0).to(self.LM.device)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            f = lambda x: self.LM.model(x)[0][..., self.LM.tokens["all"]].cpu().float()
        
            x = torch.clone(input_ids).to(self.LM.device)
            ln = x.shape[1]
            
            fx = f(x)[0]
            if self.fast:
                fx_h = torch.zeros(
                    (ln, 1 , ln, self.LM.num_tokens), 
                    dtype=torch.float32
                )
            else:
                fx_h = torch.zeros(
                    (ln, self.LM.num_tokens, ln, self.LM.num_tokens),
                    dtype=torch.float32
                )
                x = torch.tile(x, [self.LM.num_tokens, 1])
                
            for n in range(ln): # for each position
                x_h = torch.clone(x)
                if self.fast:
                    x_h[:, n] = self.LM.mask_token_id
                else:
                    x_h[:, n] = torch.tensor(self.tokens["all"])
                fx_h[n] = f(x_h)

        return fx_h, fx

    def cosine_dissimilarity(self, fx_h, fx):
        # Vectorised cosine dissimilarity computation for fast version of CJ computations
        cos = torch.nn.CosineSimilarity(dim=2)
        # NOTE: 4 is the index of the first aa token in gLM2 and ESM2
        #       It might be important to consider for other pLMs.
        jac = (torch.clamp(cos(fx_h[:, 4], fx[:, 4]), -1.0, 1.0)+1)/2
        jac = torch.ones_like(jac) - jac
        return jac

    def get_cosine_contacts(self, fx_h, fx, masks):
        fx_h_masked = self.LM.apply_masks(fx_h, masks)
        fx_masked = self.LM.apply_masks(fx, masks)

        jac = self.cosine_dissimilarity(fx_h_masked, fx_masked)

        # Symmetrisation
        contacts = (jac+jac.T)/2
        
        contacts = contacts.numpy()
        
        # Removal of diagonal values
        np.fill_diagonal(contacts, 0)
        
        return contacts

    def get_euclidean_contacts(self, fx_h, fx, masks, symm=True, center=True, diag="remove", apc=True):
        jac = fx_h - fx
        jac = self.LM.apply_masks(jac, masks)

        X = jac.numpy().copy()
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
            contacts = contacts/np.sqrt(contacts_diag[:,None]*contacts_diag[None,:])

        if apc:
            ap = contacts.sum(0,keepdims=True)*contacts.sum(1, keepdims=True)/contacts.sum()
            contacts = contacts - ap

        if diag == "remove":
            np.fill_diagonal(contacts,0)

        return contacts
    
    def get_contacts(self, sequence: str, length1: int):
        input_ids, tokens, seqlen = self.LM.get_tokenized(sequence)
        masks = self.LM.get_masks(input_ids, seqlen)
        fx_h, fx = self.get_logits(input_ids)
        if(self.distance == "Euclidean"):
            contacts = self.get_euclidean_contacts(fx_h, fx, masks)
        elif(self.distance == "cosine"):
            contacts = self.get_cosine_contacts(fx_h, fx, masks)

        return contacts 
    
    def contact_to_dataframe(self, con):
        sequence_length = con.shape[0]
        idx = [str(i) for i in np.arange(1, sequence_length+1)]
        df = pd.DataFrame(con, index=idx, columns=idx)
        df = df.stack().reset_index()
        df.columns = ['i', 'j', 'value']
        return df
    
    def is_computed(self, id):
        cj_path = pathlib.Path(f"{self.matrix_path}/{id}_{self.cj_type}CJ.npy")
        if(cj_path.is_file() and cj_path.stat().st_size != 0):
            return True

    def outlier_count(self, array, upper_right_quadrant, mode="IQR", n=3, denominator=1e-8):
        if mode == "IQR":
            Q1 = np.percentile(array, 25)
            Q3 = np.percentile(array, 75)
            IQR = Q3-Q1
            threshold = Q3+1.5*IQR

        elif mode == "mean_stddev":
            m = np.mean(array)
            s = np.std(array)
            threshold = m+n*s

        elif mode == "ratio":
            threshold = 0.7
            array /= denominator

        count_above_threshold = np.sum(upper_right_quadrant > threshold)

        return count_above_threshold
    
    def detect_ppi(self, array, len1, padding=0.1):
        # Calculate the number of residues to ignore

        if(padding < 1):
            ignore_len1 = int(len1*padding)
            ignore_len2 = int((array.shape[0]-len1)*padding)
        else:
            ignore_len1 = padding
            ignore_len2 = padding

        # Detecting the PPI signal in upper right quadrant of matrix
        upper_right_quadrant = array[ignore_len1:len1-ignore_len1, len1+ignore_len2:-ignore_len2]
        quadrant_size = upper_right_quadrant.shape[0]*upper_right_quadrant.shape[1]

        # Detect outliers
        ppi = self.outlier_count(array, upper_right_quadrant, mode="IQR", n=3)/quadrant_size

        # Just a placeholder for the counting stage
        ppi_lab = 1 if(ppi) else 0
        
        # Detecting the PPI signal in upper right quadrant of matrix
        return ppi, ppi_lab

    def apply_z_scores(self, array_2d):
        quadrant = array_2d

        z_rows = (quadrant-quadrant.mean(axis=1, keepdims=True))/quadrant.std(axis=1, keepdims=True)
        z_cols = (quadrant-quadrant.mean(axis=0, keepdims=True))/quadrant.std(axis=0, keepdims=True)
        
        array_2d = (z_rows+z_cols)/2

        return array_2d

    def forward(self, x, x_idx, stage):
        ppi_preds = []
        ppi_labs = []

        for i, s in enumerate(x['sequence']):
            if(self.is_computed(x['concat_id'][i])):
                # Load the already computed matrix
                array_2d = np.load(f"{self.matrix_path}/{x['concat_id'][i]}_{self.cj_type}CJ.npy")
            else:
                contacts = self.get_contacts(s, x['length1'][i])
                df = self.contact_to_dataframe(contacts)

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

                np.save(f"{self.matrix_path}/{x['concat_id'][i]}_{self.cj_type}CJ.npy", array_2d)

            # Detect the PPI signal in the CJ
            ppi_pred, ppi_lab = self.detect_ppi(array_2d, x['length1'][i])
            ppi_preds.append(ppi_pred) 
            ppi_labs.append(ppi_lab) 

        return torch.FloatTensor(ppi_preds), torch.IntTensor(ppi_labs)
    
    def compute_loss(self, x):
        predictions = x['predictions']
        pred_labels = x['predicted_label']
        return {'loss': zero_one_loss(x['label'].detach().cpu(), pred_labels.detach().cpu())}

