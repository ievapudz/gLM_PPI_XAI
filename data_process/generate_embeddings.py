#!/usr/bin/env python3

from optparse import OptionParser
import torch
from transformers import AutoModel, AutoTokenizer
from Processor import Processor

parser = OptionParser()

parser.add_option("--fasta", "-f", dest="fasta",
	default=None, help="path to the input FASTA file.")

parser.add_option("--pair-list", "-p", dest="pair_list",
	default=None, help="path to the pair list file (sequence identifiers of "+\
        "a pair separated with a whitespace character).")

parser.add_option("--output", "-o", dest="output",
	default=None, help="path to the output PT file with embeddings.")

(options, args) = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModel.from_pretrained('tattabio/gLM2_650M', torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M', trust_remote_code=True)

proc = Processor(options.fasta, options.pair_list)
fasta_dict = proc.load_fasta()

pairs = proc.load_pair_list()
pair_embeddings = {}

for p in pairs:
    id, seq = proc.process_pair(p, fasta_dict)
    print(id, seq)

    encodings = tokenizer([seq], return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        embeddings = model(encodings.input_ids, output_hidden_states=True).last_hidden_state
            
    pair_embeddings[id] = embeddings[0]

torch.save(pair_embeddings, options.output)