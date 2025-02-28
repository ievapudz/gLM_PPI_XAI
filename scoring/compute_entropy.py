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
import seaborn as sns

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

def get_token_labels(tokenizer, all_tokens):
    return tokenizer.convert_ids_to_tokens(all_tokens)

def get_entropy(sequence: str):
    all_tokens = NUC_TOKENS + AA_TOKENS
    num_tokens = len(all_tokens)
    input_ids = torch.tensor(tokenizer.encode(sequence), dtype=torch.int)
    input_ids = input_ids.unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
        #f = lambda x: model(x)[0][..., all_tokens].cpu().float()
        f = lambda x: model(x)[0].cpu().float()
        x = torch.clone(input_ids).to(DEVICE)
        ln = x.shape[1]

        # Retrieval of logits (shape: len x num_tokens)
        fx = f(x)[0]
        #token_labels = get_token_labels(tokenizer, all_tokens)
        token_labels = get_token_labels(tokenizer, range(0, tokenizer.vocab_size))

        # Retrieval of probabilities
        p = torch.nn.functional.softmax(fx, dim=1)
        #p.token_map = {i: (all_tokens[i], token_labels[i]) for i in range(len(token_labels))}
        p.token_map = {i: (i, token_labels[i]) for i in range(len(token_labels))}
        
        # Compute entropy
        entropy = Categorical(probs=p).entropy()
        max_entropy = Categorical(probs=torch.FloatTensor([1/num_tokens]*num_tokens)).entropy()

        return entropy/max_entropy, p

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

def plot_probabilities(probabilities, id):
    token_map = probabilities.token_map
    token_labels = [token_map[i][1] for i in range(len(token_map))]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(probabilities.T, cmap='viridis', cbar=True, yticklabels=token_labels, vmin=0, vmax=1)
    plt.xlabel('Position in Protein Sequence')
    plt.ylabel('Token')
    plt.title(f"{id} probabilities of tokens")
    plt.yticks(rotation=0) 
    plt.savefig(f'outputs/visualisations/probabilities/{id}.png')
    plt.close()

def plot_combined(entropy_values, probabilities, id):
    token_map = probabilities.token_map
    token_labels = [token_map[i][1] for i in range(len(token_map))]
    positions = np.arange(0, entropy_values.shape[0])

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Create a second y-axis for the heatmap
    ax2 = ax1.twinx()
    ax2.set_ylabel('Token')
    im = ax2.imshow(probabilities.T, aspect='auto', cmap='viridis', vmin=0, vmax=1, alpha=0.7)
    fig.colorbar(im, ax=ax2)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_yticks(np.arange(len(token_labels)) + 0.5)
    ax2.set_yticklabels(token_labels, rotation=0)

    # Plot entropy
    ax1.set_xlabel('Position in Protein Sequence')
    ax1.set_ylabel('Normalized Entropy')
    ax1.plot(positions, entropy_values, marker=None, linestyle='-', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, 1)

    plt.title(f"{id} Combined Entropy and Probabilities")
    fig.tight_layout()
    plt.savefig(f'outputs/visualisations/combined/{id}.png')
    plt.close()

parser = OptionParser()

parser.add_option("--input", "-i", dest="input",
	default=None,  help="protein ids to process.")

parser.add_option("--complexes", "-c", dest="are_complexes",
	default=False, action="store_true", help="flag to compute the entropy for "+
    "protein complexes.")

parser.add_option("--probabilities", "-p", dest="probs",
	default=False, action="store_true", help="flag to compute and plot the "+
    "probabilities instead of entropy.")

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

    norm_entropy, probs = get_entropy(sequence)
    if(options.probs):
        #plot_probabilities(probs, id)
        plot_combined(norm_entropy, probs, id)
    else:
        plot_entropy(norm_entropy, id)

