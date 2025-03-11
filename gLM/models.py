from pytorch_lightning import LightningModule
from torch import nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import pandas as pd
import math
from sklearn.metrics import zero_one_loss
from pytorch_lightning import Trainer
import pathlib
from scipy.ndimage import gaussian_filter1d
from torch.distributions import Categorical
import scipy.ndimage as ndimage

class CategoricalJacobian(nn.Module):
    def __init__(self, fast: bool, matrix_path: str):
        super().__init__()
        self.fast = fast
        self.cj_type = 'fast' if(self.fast) else 'full'

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

        self.matrix_path = matrix_path
        pathlib.Path(self.matrix_path).mkdir(parents=True, exist_ok=True)

        self.too_long = []

    def conditioned_APC(self, s, len1):
        new_s = np.zeros(np.shape(s))

        for i in range(len(s)):
            for j in range(len(s)):
                if i < len1 and j < len1:
                    # Case when both residues belong to the first protein
                    expected = np.sum(s[i, :len1]) * np.sum(s[:len1, j]) / np.sum(s[:len1, :len1])
                    new_s[i, j] = s[i, j] - expected
                elif i >= len1 and j >= len1:
                    # Case when both residues belong to the second protein
                    expected = np.sum(s[i, len1:]) * np.sum(s[len1:, j]) / np.sum(s[len1:, len1:])
                    new_s[i, j] = s[i, j] - expected
                else:
                    # Case when residues come from different proteins (ensure symmetry)
                    E_ij = np.sum(s[i, len1:]) * np.sum(s[:len1, j]) / np.sum(s[len1:, :len1])
                    E_ji = np.sum(s[j, len1:]) * np.sum(s[:len1, i]) / np.sum(s[len1:, :len1])
                    expected = (E_ij + E_ji) / 2  # Symmetric correction
                    new_s[i, j] = new_s[j, i] = s[i, j] - expected

        return new_s

    def jac_to_contact(self, jac, length1, symm=True, center=True, diag="remove", apc=True):
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
            contacts = contacts/np.sqrt(contacts_diag[:,None]*contacts_diag[None,:])

        if apc:
            ap = contacts.sum(0,keepdims=True)*contacts.sum(1, keepdims=True)/contacts.sum()
            contacts = contacts - ap

        if diag == "remove":
            np.fill_diagonal(contacts,0)

        return contacts

    def get_categorical_jacobian(self, sequence: str, length1: int):
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
                fx_h = torch.zeros((ln, 1, ln, num_tokens), dtype=torch.float32)
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
            contact = self.jac_to_contact(jac.numpy(), length1)
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
    
    def is_computed(self, id):
        # TODO: adjust the naming of the output files after debugging
        cj_path = pathlib.Path(f"{self.matrix_path}/{id}_{self.cj_type}CJ.npy")
        if(cj_path.is_file() and cj_path.stat().st_size != 0):
            return True

    def outlier_count(self, upper_right_quadrant, mode="IQR", n=3, denominator=1e-8):
        if mode == "IQR":
            Q1 = np.percentile(upper_right_quadrant, 25)
            Q3 = np.percentile(upper_right_quadrant, 75)
            IQR = Q3-Q1
            threshold = Q3+1.5*IQR

        elif mode == "mean_stddev":
            m = np.mean(upper_right_quadrant)
            s = np.std(upper_right_quadrant)
            threshold = m+n*s

        elif mode == "ratio":
            threshold = 0.7
            upper_right_quadrant /= denominator

        count_above_threshold = np.sum(upper_right_quadrant > threshold)

        return count_above_threshold
    
    def detect_ppi(self, array, len1, padding=0.1):
        # Calculate the number of residues to ignore
        ignore_len1 = int(len1*padding)
        ignore_len2 = int((array.shape[0]-len1)*padding)

        # Detecting the PPI signal in upper right quadrant of matrix
        upper_right_quadrant = array[ignore_len1:len1-ignore_len1, len1+ignore_len2:-ignore_len2]
        quadrant_size = upper_right_quadrant.shape[0]*upper_right_quadrant.shape[1]

        # Detect outliers
        ppi = self.outlier_count(upper_right_quadrant, mode="mean_stddev", n=3)/quadrant_size

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

    def apply_patching(self, array_2d, len1):
        quadrant = array_2d[:len1, len1:]
        gaussian_filtered = ndimage.gaussian_filter(quadrant, sigma=1)
        mean_filtered = ndimage.uniform_filter(gaussian_filtered, size=5)
        array_2d[:len1, len1:] = mean_filtered

        return array_2d

    def forward(self, x):
        sigmoid_v = np.vectorize(self.sigmoid)
        ppi_preds = []
        ppi_labs = []

        for i, s in enumerate(x['sequence']):
            # TODO: has to be debugged properly - so that the number of true
            #       labels would correspond to the number of done predictions
            #if(len(x['sequence'][i]) > 1000):
            #    print(x['concat_id'][i])
            #    continue

            if(self.is_computed(x['concat_id'][i])):
                # Load the already computed matrix
                array_2d = np.load(f"{self.matrix_path}/{x['concat_id'][i]}_{self.cj_type}CJ.npy")
            else:
                J, contact, tokens = self.get_categorical_jacobian(s, x['length1'][i])
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
                np.save(f"{self.matrix_path}/{x['concat_id'][i]}_{self.cj_type}CJ.npy", array_2d)

            # Detect the PPI signal in the CJ
            array_2d = self.apply_z_scores(array_2d, length1)
            array_2d = self.apply_patching(array_2d, length1)
            ppi_pred, ppi_lab = self.detect_ppi(array_2d, x['length1'][i])
            ppi_preds.append(ppi_pred) 
            ppi_labs.append(ppi_lab) 

        return torch.FloatTensor(ppi_preds), torch.IntTensor(ppi_labs)
    
    def compute_loss(self, x):
        predictions = x['predictions']
        pred_labels = x['predicted_label']
        return {'loss': zero_one_loss(x['label'].detach().cpu(), pred_labels.detach().cpu())}

class MutationEntropyMatrix(nn.Module):
    def __init__(self, fast: bool, matrix_path: str):
        super().__init__()
        self.fast = fast
        self.type = 'fast' if(self.fast) else 'full'

        self.nuc_tokens = tuple(range(29, 33)) # 4 nucleotides a,t,c,g
        self.aa_tokens = tuple(range(4, 24)) # 20 amino acids

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_path = "./gLM2_650M"
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.MASK_TOKEN_ID = self.tokenizer.mask_token_id

        for param in self.model.parameters():
            param.requires_grad = False

        self.matrix_path = matrix_path
        pathlib.Path(self.matrix_path).mkdir(parents=True, exist_ok=True)

    def is_computed(self, id):
        m_path = pathlib.Path(f"{self.matrix_path}/{id}_{self.type}Entropy.npy")
        if(m_path.is_file() and m_path.stat().st_size != 0):
            return True

    def outlier_count(self, upper_right_quadrant, mode="IQR", n=3):
        if mode == "IQR":
            Q1 = np.percentile(upper_right_quadrant, 25)
            Q3 = np.percentile(upper_right_quadrant, 75)
            IQR = Q3-Q1
            threshold = Q3+1.5*IQR

        elif mode == "mean_stddev":
            m = np.mean(upper_right_quadrant)
            s = np.std(upper_right_quadrant)
            threshold = m+n*s

        elif mode == "P95":
            threshold = max(np.percentile(upper_right_quadrant, 95), 0.1)

        count_above_threshold = np.sum(upper_right_quadrant > threshold)

        return count_above_threshold
    
    def detect_ppi(self, array, len1, padding=0.1):
        if(padding == 0):
            upper_right_quadrant = array[:len1, len1:]
        else:
            # Calculate the number of residues to ignore
            ignore_len1 = int(len1*padding)
            ignore_len2 = int((array.shape[0]-len1)*padding)

            # Detecting the PPI signal in upper right quadrant of matrix
            upper_right_quadrant = array[ignore_len1:len1-ignore_len1, len1+ignore_len2:-ignore_len2]

        quadrant_size = upper_right_quadrant.shape[0]*upper_right_quadrant.shape[1]

        # Detect outliers
        ppi = self.outlier_count(upper_right_quadrant, mode="P95")/quadrant_size

        # Just a placeholder for the counting stage
        ppi_lab = 1 if(ppi) else 0
        
        # Detecting the PPI signal in upper right quadrant of matrix
        return ppi, ppi_lab

    def get_matrix(self, sequence: str, length1: int):
        all_tokens = self.nuc_tokens + self.aa_tokens
        num_tokens = len(all_tokens)

        input_ids = torch.tensor(self.tokenizer.encode(sequence), dtype=torch.int)

        input_ids = input_ids.unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
            f = lambda x: self.model(x)[0][..., all_tokens].cpu().float()

            x = torch.clone(input_ids).to(self.device)
            ln = x.shape[1]

            fx = f(x)[0]
            if self.fast:
                fx_h = torch.zeros((ln, 1, ln, num_tokens), dtype=torch.float32)
            else:
                fx_h = torch.zeros((ln, num_tokens, ln, num_tokens), dtype=torch.float32)
                x = torch.tile(x, [num_tokens, 1])

            for n in range(ln): # for each position
                x_h = torch.clone(x)
                if self.fast:
                    x_h[:, n] = self.MASK_TOKEN_ID
                else:
                    x_h[:, n] = torch.tensor(all_tokens)
                fx_h[n] = f(x_h)

            probx_h = torch.nn.functional.softmax(fx_h, dim=-1)
            probx = torch.nn.functional.softmax(fx, dim=1)

            entropy_h = Categorical(probs=probx_h).entropy()
            entropy = Categorical(probs=probx).entropy()

            max_entropy = Categorical(probs=torch.FloatTensor([1/num_tokens]*num_tokens)).entropy()

            delta = (entropy_h - entropy)/max_entropy
            
        return delta

    def forward(self, x):
        ppi_preds = []
        ppi_labs = []

        for i, s in enumerate(x['sequence']):
            if(self.is_computed(x['concat_id'][i])):
                # Load the already computed matrix
                entropy_m = np.load(f"{self.matrix_path}/{x['concat_id'][i]}_{self.type}Entropy.npy")
            else:
                entropy_m = self.get_matrix(s, x['length1'][i])
                entropy_m = entropy_m.cpu().detach().numpy().squeeze(1)
                np.save(f"{self.matrix_path}/{x['concat_id'][i]}_{self.type}Entropy.npy", entropy_m)

            # Detect the PPI signal in the entropy matrix
            ppi_pred, ppi_lab = self.detect_ppi(entropy_m, x['length1'][i], padding=0.1)
            ppi_preds.append(ppi_pred) 
            ppi_labs.append(ppi_lab) 

        return torch.FloatTensor(ppi_preds), torch.IntTensor(ppi_labs)

    def compute_loss(self, x):
        predictions = x['predictions']
        pred_labels = x['predicted_label']
        return {'loss': zero_one_loss(x['label'].detach().cpu(), pred_labels.detach().cpu())}


class EntropyFactors(nn.Module):
    def __init__(self, matrix_path: str):
        super().__init__()
        self.nuc_tokens = tuple(range(29, 33)) # 4 nucleotides a,t,c,g
        self.aa_tokens = tuple(range(4, 24)) # 20 amino acids

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_path = "./gLM2_650M"
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.MASK_TOKEN_ID = self.tokenizer.mask_token_id

        for param in self.model.parameters():
            param.requires_grad = False

        self.matrix_path = matrix_path

    def is_computed(self, id):
        m_path = pathlib.Path(f"{self.matrix_path}/{id}_EntropyFactors.npy")
        if(m_path.is_file() and m_path.stat().st_size != 0):
            return True

    def forward(self, x):
        ppi_preds = []
        ppi_labs = []

        for i, s in enumerate(x['sequence']):
            # Get entropy of the sequence
            seq_entropy = self.get_entropy(s)
            ppi_pred, ppi_lab = self.average_entropy(seq_entropy)
            ppi_preds.append(ppi_pred) 
            ppi_labs.append(ppi_lab)

            if(self.is_computed(x['concat_id'][i])):
                # Load the already computed matrix
                entropy_f = np.load(f"{self.matrix_path}/{x['concat_id'][i]}_EntropyFactors.npy")
            else:
                # Get entropy factors
                entropy_f = self.get_matrix(seq_entropy)
                np.save(f"{self.matrix_path}/{x['concat_id'][i]}_EntropyFactors.npy", entropy_f)
            
        return torch.FloatTensor(ppi_preds), torch.IntTensor(ppi_labs)
    
    def get_entropy(self, sequence: str):
        all_tokens = self.nuc_tokens + self.aa_tokens
        num_tokens = len(all_tokens)

        input_ids = torch.tensor(self.tokenizer.encode(sequence), dtype=torch.int)
        input_ids = input_ids.unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
            f = lambda x: self.model(x)[0][..., all_tokens].cpu().float()

            x = torch.clone(input_ids).to(self.device)
            fx = f(x)[0]

            probx = torch.nn.functional.softmax(fx, dim=1)
            entropy = Categorical(probs=probx).entropy()
            max_entropy = Categorical(probs=torch.FloatTensor([1/num_tokens]*num_tokens)).entropy()
            entropy = (entropy)/max_entropy
            
        return entropy

    def average_entropy(self, seq_entropy):
        # Compute average entropy
        ppi = np.average(seq_entropy)

        # Just a placeholder 
        ppi_lab = 0
        
        # Detecting the PPI signal in upper right quadrant of matrix
        return ppi, ppi_lab

    def get_matrix(self, seq_entropy):
        return 1 - np.outer(np.array(seq_entropy), np.array(seq_entropy))

    def compute_loss(self, x):
        predictions = x['predictions']
        pred_labels = x['predicted_label']
        return {'loss': zero_one_loss(x['label'].detach().cpu(), pred_labels.detach().cpu())}

class EmbeddingsMatrix(nn.Module):
    def __init__(self, fast: bool, matrix_path: str):
        super().__init__()
        # TODO: check if these could be removed
        self.nuc_tokens = tuple(range(29, 33)) # 4 nucleotides a,t,c,g
        self.aa_tokens = tuple(range(4, 24)) # 20 amino acids
        self.emb_dim = 1280
        self.fast = fast
        self.type = 'fast' if(self.fast) else 'full'

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_path = "./gLM2_650M"
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.MASK_TOKEN_ID = self.tokenizer.mask_token_id

        for param in self.model.parameters():
            param.requires_grad = False

        self.matrix_path = matrix_path
        pathlib.Path(self.matrix_path).mkdir(parents=True, exist_ok=True)

    def outlier_count(self, upper_right_quadrant, mode="IQR", n=3):
        if mode == "IQR":
            Q1 = np.percentile(upper_right_quadrant, 25)
            Q3 = np.percentile(upper_right_quadrant, 75)
            IQR = Q3-Q1
            threshold = Q3+1.5*IQR

        elif mode == "mean_stddev":
            m = np.mean(upper_right_quadrant)
            s = np.std(upper_right_quadrant)
            threshold = m+n*s

        elif mode == "P95":
            threshold = max(np.percentile(upper_right_quadrant, 95), 0.1)

        count_above_threshold = np.sum(upper_right_quadrant > threshold)

        return count_above_threshold

    def detect_ppi(self, array, len1, padding=0.1):
        # Calculate the number of residues to ignore
        ignore_len1 = int(len1*padding)
        ignore_len2 = int((array.shape[0]-len1)*padding)

        # Detecting the PPI signal in upper right quadrant of matrix
        upper_right_quadrant = array[ignore_len1:len1-ignore_len1, len1+ignore_len2:-ignore_len2]
        quadrant_size = upper_right_quadrant.shape[0]*upper_right_quadrant.shape[1]

        # Detect outliers
        ppi = self.outlier_count(upper_right_quadrant, mode="IQR")/quadrant_size

        # Just a placeholder for the counting stage
        ppi_lab = 0
        
        # Detecting the PPI signal in upper right quadrant of matrix
        return ppi, ppi_lab

    def jac_to_contact(self, emb_jac, symm=True, center=True, diag="remove", apc=True):
        X = emb_jac.copy()
        Lx, Ax, Ly, Ay = X.shape

        if center:
            for i in range(4):
                if X.shape[i] > 1:
                    X -= X.mean(i, keepdims=True)

        contacts = np.sqrt(np.square(X).sum((1,3)))

        if symm and (Ax != self.emb_dim or Ay != self.emb_dim):
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

    def get_matrix(self, sequence: str, length1: int):
        # TODO: check if these could be removed
        all_tokens = self.nuc_tokens + self.aa_tokens
        num_tokens = len(all_tokens)

        input_ids = torch.tensor(self.tokenizer.encode(sequence), dtype=torch.int)

        input_ids = input_ids.unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
            f = lambda x: self.model(x, output_hidden_states=True).hidden_states[-1].cpu().float()

            x = torch.clone(input_ids).to(self.device)
            ln = x.shape[1]

            fx = f(x)[0]
            if self.fast:
                fx_h = torch.zeros((ln, 1, ln, self.emb_dim), dtype=torch.float32)
            else:
                fx_h = torch.zeros((ln, self.emb_dim, ln, self.emb_dim), dtype=torch.float32)
                x = torch.tile(x, [emb_dim, 1])

            for n in range(ln): # for each position
                x_h = torch.clone(x)
                if self.fast:
                    x_h[:, n] = self.MASK_TOKEN_ID
                else:
                    x_h[:, n] = torch.tensor(all_tokens)
                fx_h[n] = f(x_h)

            emb_jac = fx_h-fx

            contact = self.jac_to_contact(emb_jac.numpy())

        return emb_jac, contact

    def contact_to_dataframe(self, con):
        sequence_length = con.shape[0]
        idx = [str(i) for i in np.arange(1, sequence_length+1)]
        df = pd.DataFrame(con, index=idx, columns=idx)
        df = df.stack().reset_index()
        df.columns = ['i', 'j', 'value']
        return df

    def is_computed(self, id):
        m_path = pathlib.Path(f"{self.matrix_path}/{id}_{self.type}Emb.npy")
        if(m_path.is_file() and m_path.stat().st_size != 0):
            return True

    def forward(self, x):
        ppi_preds = []
        ppi_labs = []

        for i, s in enumerate(x['sequence']):
            if(self.is_computed(x['concat_id'][i])):
                # Load the already computed matrix
                emb_m = np.load(f"{self.matrix_path}/{x['concat_id'][i]}_{self.type}Emb.npy")
            else:
                emb_m, contact = self.get_matrix(s, x['length1'][i])
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
                emb_m = pivot_df.to_numpy()
                np.save(f"{self.matrix_path}/{x['concat_id'][i]}_{self.type}Emb.npy", emb_m)

            ppi_pred, ppi_lab = self.detect_ppi(emb_m, x['length1'][i])

            ppi_preds.append(ppi_pred) 
            ppi_labs.append(ppi_lab)

        return torch.FloatTensor(ppi_preds), torch.IntTensor(ppi_labs)

    def compute_loss(self, x):
        predictions = x['predictions']
        pred_labels = x['predicted_label']
        return {'loss': zero_one_loss(x['label'].detach().cpu(), pred_labels.detach().cpu())}

class gLM2(LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.save_hyperparameters()
        
    def step(self, batch, batch_idx, split):
        batch['predictions'], batch['predicted_label'] = self.model(batch)
    
        self.step_outputs[split] = batch
        losses = self.model.compute_loss(batch)
        for key, value in losses.items():
            self.log(f'{split}/{key}', value)
        return losses['loss']

    def training_step(self, batch, batch_idx):
        self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx, 'test')

