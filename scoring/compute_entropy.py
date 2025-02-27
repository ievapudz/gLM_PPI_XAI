#!/usr/bin/env python3

# Computation of entropy based on logits from gLMs at each position of the protein

from optparse import OptionParser
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from data_process import Processor

NUC_TOKENS = tuple(range(29, 33)) # 4 nucleotides a,t,c,g
AA_TOKENS = tuple(range(4, 24)) # 20 amino acid

MODEL_PATH = "./gLM2_650M"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, trust_remote_code=True).eval().to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

def get_sequences(sequence_ids, fasta_file):
    sequences = {}

    with open(fasta_file, "r") as fh:
        for record in SeqIO.parse(fh, "fasta"):
            if(record.id in sequence_ids): sequences[record.id] = f"<+>{record.seq}"

    return sequences

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
        p = torch.nn.functional.softmax(fx, dim=1)

        # Compute entropy
        entropy = Categorical(probs=p).entropy()
        max_entropy = Categorical(probs=torch.FloatTensor([1/num_tokens]*num_tokens)).entropy()

        return entropy/max_entropy

def plot_entropy(entropy_values, id):
    positions = np.arange(0, entropy_values.shape[0])
    plt.figure(figsize=(10, 6))
    plt.plot(positions, entropy_values, marker=None, linestyle='-', color='b')
    plt.xlabel('Position in Protein Sequence')
    plt.ylabel('Normalized Entropy')
    plt.title(f"{id} entropy at each position")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'outputs/visualisations/entropy/{id}.png')
    plt.close()

parser = OptionParser()

parser.add_option("--input", "-i", dest="input",
	default=None,  help="protein ids to process.")

parser.add_option("--complexes", "-c", dest="are_complexes",
	default=False, action="store_true", help="flag to compute the entropy for protein complexes.")

parser.add_option("--fasta", "-f", dest="fasta",
	default="./data/Bernett2022/human_swissprot_oneliner.fasta",  
    help="FASTA with sequences.")

(options, args) = parser.parse_args()

with open(options.input, 'r') as file:
    options.input = [line.strip() for line in file.readlines()]

if(options.are_complexes):
    proc = Processor.Processor(fasta_path=options.fasta, pair_list_path=options.input)
    fasta = proc.load_fasta()
    sequences = {}
    for pair in options.input:
        print(pair)
        _, concat_seq, _, _ = proc.process_pair(pair, fasta, aa_only=True, ready_pair_ids=True)
        sequences[pair] = concat_seq
else:
    sequences = get_sequences(options.input, options.fasta)

for id in options.input: 
    sequence = sequences[id]

    norm_entropy = get_entropy(sequence)
    plot_entropy(norm_entropy, id)

