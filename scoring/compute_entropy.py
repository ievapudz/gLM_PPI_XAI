#!/usr/bin/env python3

# Computation of entropy based on logits from gLMs at each position of the protein

from optparse import OptionParser
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

NUC_TOKENS = tuple(range(29, 33)) # 4 nucleotides a,t,c,g
AA_TOKENS = tuple(range(4, 24)) # 20 amino acid

MODEL_PATH = "./gLM2_650M"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, trust_remote_code=True).eval().to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

def get_entropy(sequence: str):
    all_tokens = NUC_TOKENS + AA_TOKENS
    num_tokens = len(all_tokens)
    input_ids = torch.tensor(tokenizer.encode(sequence), dtype=torch.int)
    input_ids = input_ids.unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
        f = lambda x:model(x)[0][..., all_tokens].cpu().float()
        x = torch.clone(input_ids).to(DEVICE)
        ln = x.shape[1]

        # Retrieval of logits (shape: len x num_tokens)
        fx = f(x)[0]

        # Retrieval of probabilities
        fx = torch.nn.functional.softmax(fx, dim=1)

        # Develop. check
        print(torch.sum(fx, dim=1))


parser = OptionParser()

parser.add_option("--input", "-i", dest="input",
	default=None,  help="input list of protein ids.")

(options, args) = parser.parse_args()

# TODO: make programmable. For development, sequence P40429 taken
sequence = f"<+>MAEVQVLVLDGRGHLLGRLAAIVAKQVLLGRKVVVVRCEGINISGNFYRNKLKYLAFLRKRMNTNPSRGPYHFRAPSRIFWRTVRGMLPHKTKRGQAALDRLKVFDGIPPPYDKKKRMVVPAALKVVRLKPTRKFAYLGRLAHEVGWKYQAVTATLEEKRKEKAKIHYRKKKQLMRLRKQAEKNVEKKIDKYTEVLKTHGLLV"

get_entropy(sequence)
