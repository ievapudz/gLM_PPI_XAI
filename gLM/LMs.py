from torch import nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from mint.mint.helpers.extract import MINTWrapper, load_config, CollateFn

class BioLM(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mask_token_id = self.tokenizer.mask_token_id
        # Tokens will be initialised in the inheriting classes
        self.tokens = {"all": []}
        self.num_tokens = 0
        self.logits_key = 0

    def is_pos_token(self, input_ids, seqlen, token_type="aa"):
        device = input_ids.device
        is_pos = torch.isin(input_ids, torch.tensor(self.tokens[token_type]))
        is_pos = is_pos.view(-1, 1, 1, 1).expand(seqlen, 1, seqlen, 1)
        is_token = torch.ones(1, 1, 1, self.num_tokens, dtype=torch.bool, 
            device=device)
        return is_pos, is_token     

    def get_masks(self):
        pass

    def apply_masks(self):
        pass

    def get_tokenized(self, sequence):
        input_ids = torch.tensor(self.tokenizer.encode(sequence), dtype=torch.int)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        seqlen = input_ids.shape[0]
        
        return input_ids, tokens, seqlen, None


    def get_logits(self, input_ids, chain_mask=None, fast=True, context_idx=None):

        input_ids = input_ids.unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            f = lambda x: self.model(x)[0][..., self.tokens["all"]].cpu().float()
            x = torch.clone(input_ids).to(self.device)
            
            if(context_idx):
                x = x[:, context_idx[0]:context_idx[1]]

            ln = x.shape[1]

            fx = f(x)[0]
            if fast:
                fx_h = torch.zeros(
                    (ln, 1 , ln, self.num_tokens),
                    dtype=torch.float32
                )
            else:
                fx_h = torch.zeros(
                    (ln, self.num_tokens, ln, self.num_tokens),
                    dtype=torch.float32
                )
                x = torch.tile(x, [self.num_tokens, 1])

            for n in range(ln): # for each position
                x_h = torch.clone(x)
                if fast:
                    x_h[:, n] = self.mask_token_id
                else:
                    x_h[:, n] = torch.tensor(self.tokens["all"])
                fx_h[n] = f(x_h)

        return fx_h, fx

class gLM2(BioLM):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.tokens = {
            "aa": tuple(range(4, 24)),
            "nuc": tuple(range(29, 33))
        }
        self.tokens["all"] = self.tokens["aa"] + self.tokens["nuc"]
        self.num_tokens = len(self.tokens["all"])

    def is_pos_token(self, input_ids, seqlen, token_type="aa"):
        device = input_ids.device
        is_pos = torch.isin(input_ids, torch.tensor(self.tokens[token_type]))
        is_pos = is_pos.view(-1, 1, 1, 1).expand(seqlen, 1, seqlen, 1)
        is_token = torch.isin(torch.tensor(self.tokens["all"]), torch.tensor(self.tokens[token_type], device=device))
        is_token = is_token.view(1, 1, 1, self.num_tokens).expand(1, 1, seqlen, self.num_tokens)
        return is_pos, is_token

    def get_masks(self, input_ids, seqlen, fast=True):
        if(fast):
            is_aa_pos, is_aa_token = self.is_pos_token(
                input_ids, seqlen, token_type="aa"
            )
            is_nuc_pos, is_nuc_token = self.is_pos_token(
                input_ids, seqlen, token_type="nuc"
            )
        else:
            device = input_ids.device
            is_nuc_pos = torch.isin(input_ids, torch.tensor(self.tokens["nuc"]))
            is_nuc_pos = is_nuc_pos.view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
            
            is_nuc_token = torch.isin(torch.tensor(self.tokens["all"]), 
                torch.tensor(self.tokens["nuc"])
            )
            is_nuc_token = is_nuc_token.view(1, -1, 1, 1)
            is_nuc_token = is_nuc_token.repeat(1, 1, 1, len(self.tokens["all"]))
            
            is_aa_pos = torch.isin(input_ids, torch.tensor(self.tokens["aa"]))
            is_aa_pos = is_aa_pos.view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
            
            is_aa_token = torch.isin(torch.tensor(self.tokens["all"]), 
                torch.tensor(self.tokens["aa"])
            )
            is_aa_token = is_aa_token.view(1, -1, 1, 1)
            is_aa_token = is_aa_token.repeat(1, 1, 1, len(self.tokens["all"]))
       
        return is_nuc_pos, is_nuc_token, is_aa_pos, is_aa_token

    def apply_masks(self, matrix, masks):
        is_nuc_pos, is_nuc_token, is_aa_pos, is_aa_token = masks 

        valid_nuc = is_nuc_pos & is_nuc_token
        valid_aa = is_aa_pos & is_aa_token
   
        # The logits' shape is just expanded
        matrix_masked = torch.where(valid_nuc | valid_aa, matrix, 0.0)

        return matrix_masked

class ESM2(BioLM):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.tokens = {
            "aa": tuple(range(4, 29))
        }
        self.tokens["all"] = self.tokens["aa"]
        self.num_tokens = len(self.tokens["all"])

    def get_masks(self, input_ids, seqlen, fast=True):
        if(fast):
            is_aa_pos, is_aa_token = self.is_pos_token(
                input_ids, seqlen, token_type="aa"
            )
        else:
            # [seqlen, 1, seqlen, 1].
            is_aa_pos = torch.isin(input_ids, 
                torch.tensor(self.tokens["aa"])
            ).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
            # [1, num_tokens, 1, num_tokens].
            is_aa_token = torch.isin(torch.tensor(self.tokens["aa"]), 
                torch.tensor(self.tokens["aa"])
            ).view(1, -1, 1, 1).repeat(1, 1, 1, len(self.tokens["aa"]))
        
        return is_aa_pos, is_aa_token

    def apply_masks(self, matrix, masks):
        is_aa_pos, is_aa_token = masks

        valid_aa = is_aa_pos & is_aa_token

        # The logits' shape is just expanded
        matrix_masked = torch.where(valid_aa, matrix, 0.0)

        return matrix_masked

class MINT(nn.Module):
    def __init__(self, model_path: str, config_path: str, sep_chains: bool):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        config = load_config(config_path)
        wrapper = MINTWrapper(config, model_path, sep_chains=True, device=self.device)
        self.tokenizer = CollateFn(2560)

        self.model = wrapper.model
        self.logits_key = "logits"

        self.tokens = {
            "aa": tuple(range(4, 29))
        }
        self.tokens["all"] = self.tokens["aa"]
        self.num_tokens = len(self.tokens["all"])
        self.mask_token_id = self.tokenizer.alphabet.mask_idx

        self.sep_chains = sep_chains

    def get_tokenized(self, sequence):
        sequence = sequence.split("<eos>")
        input_ids, chain_masks = self.tokenizer([sequence])
        tokens = [self.tokenizer.alphabet.get_tok(i) for i in input_ids[0]]
        seqlen = input_ids.shape[1]

        return input_ids, tokens, seqlen, chain_masks

    def is_pos_token(self, input_ids, seqlen, token_type="aa"):
        device = input_ids.device
        is_pos = torch.isin(input_ids, torch.tensor(self.tokens[token_type]))
        is_pos = is_pos.view(-1, 1, 1, 1).expand(seqlen, 1, seqlen, 1)
        is_token = torch.ones(1, 1, 1, self.num_tokens, dtype=torch.bool,
            device=device)
        return is_pos, is_token

    def get_masks(self, input_ids, seqlen, fast=True):
        if(fast):
            is_aa_pos, is_aa_token = self.is_pos_token(
                input_ids, seqlen, token_type="aa"
            )
        else:
            # [seqlen, 1, seqlen, 1].
            is_aa_pos = torch.isin(input_ids,
                torch.tensor(self.tokens["aa"])
            ).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
            # [1, num_tokens, 1, num_tokens].
            is_aa_token = torch.isin(torch.tensor(self.tokens["aa"]),
                torch.tensor(self.tokens["aa"])
            ).view(1, -1, 1, 1).repeat(1, 1, 1, len(self.tokens["aa"]))

        return is_aa_pos, is_aa_token

    def apply_masks(self, matrix, masks):
        is_aa_pos, is_aa_token = masks

        valid_aa = is_aa_pos & is_aa_token

        # The logits' shape is just expanded
        matrix_masked = torch.where(valid_aa, matrix, 0.0)

        return matrix_masked

    def get_logits(self, input_ids, chain_mask=None, fast=True):
        input_ids = input_ids.to(self.device)
        if(self.sep_chains):
            chain_mask = chain_mask.to(self.device)
        else:
            chain_mask = None

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            f = lambda x, y: self.model(x, y)["logits"][..., self.tokens["all"]].cpu().float()
            x = torch.clone(input_ids).to(self.device)
            ln = x.shape[1]

            fx = f(x, chain_mask)[0]
            if fast:
                fx_h = torch.zeros(
                    (ln, 1, ln, self.num_tokens),
                    dtype=torch.float32
                )
            else:
                fx_h = torch.zeros(
                    (ln, self.num_tokens, ln, self.num_tokens),
                    dtype=torch.float32
                )
                x = torch.tile(x, [self.num_tokens, 1])

            for n in range(ln): # for each position
                x_h = torch.clone(x)
                if fast:
                    x_h[:, n] = self.mask_token_id
                else:
                    x_h[:, n] = torch.tensor(self.tokens["all"])
                fx_h[n] = f(x_h, chain_mask)

        return fx_h, fx
