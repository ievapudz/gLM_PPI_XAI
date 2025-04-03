from torch import nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from mint.helpers.extract import MINTWrapper, load_config, CollateFn

class BioLM(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mask_token_id = self.tokenizer.mask_token_id
        # Tokens will be initialised in the inheriting classes
        self.tokens = {}
        self.logits_key = 0

    def get_masks(self):
        pass

    def apply_masks(self):
        pass

    def get_tokenized(self, sequence):
        input_ids = torch.tensor(self.tokenizer.encode(sequence), dtype=torch.int)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        seqlen = input_ids.shape[0]
        
        return input_ids, tokens, seqlen

class gLM2(BioLM):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.tokens = {
            "aa": tuple(range(4, 24)),
            "nuc": tuple(range(29, 33))
        }
        self.tokens["all"] = self.tokens["aa"] + self.tokens["nuc"]
        self.num_tokens = len(self.tokens["all"])

    def get_masks(self, input_ids, seqlen):
        # [seqlen, 1, seqlen, 1].
        is_nuc_pos = torch.isin(input_ids, 
            torch.tensor(self.tokens["nuc"])
        ).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
        # [1, num_tokens, 1, num_tokens].
        is_nuc_token = torch.isin(torch.tensor(self.tokens["all"]), 
            torch.tensor(self.tokens["nuc"])
        ).view(1, -1, 1, 1).repeat(1, 1, 1, len(self.tokens["all"]))
        # [seqlen, 1, seqlen, 1].
        is_aa_pos = torch.isin(input_ids, 
            torch.tensor(self.tokens["aa"])
        ).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
        # [1, num_tokens, 1, num_tokens].
        is_aa_token = torch.isin(torch.tensor(self.tokens["all"]), 
            torch.tensor(self.tokens["aa"])
        ).view(1, -1, 1, 1).repeat(1, 1, 1, len(self.tokens["all"]))
        
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

    def get_masks(self, input_ids, seqlen):
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
    def __init__(self, model_path: str, config_path: str):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        config = load_config(config_path)
        wrapper = MINTWrapper(config, model_path, sep_chains=True, device=self.device)
        self.tokenizer = CollateFn(1024)

        self.model = wrapper.model
        self.logits_key = "logits"

        self.tokens = {
            "aa": tuple(range(4, 29))
        }
        self.tokens["all"] = self.tokens["aa"]
        self.num_tokens = len(self.tokens["all"])
        self.mask_token_id = self.tokenizer.alphabet.mask_idx

    def get_tokenized(self, sequences):
        input_ids, chain_ids = self.tokenizer([sequences])
        tokens = []
        for j in range(len(input_ids)):
            tokens.append([self.tokenizer.alphabet.get_tok(i) for i in input_ids[j]])
        seqlen = input_ids.shape[1]

        return input_ids, tokens, seqlen

    def get_masks(self, input_ids, seqlen):
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
