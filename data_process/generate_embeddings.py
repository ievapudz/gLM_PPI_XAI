#!/usr/bin/env python3

from Processor import Processor
import torch
from transformers import AutoModel, AutoTokenizer

proc = Processor("../data/Bernett2022/human_swissprot_oneliner.fasta", "../data/Bernett2022/Intra2_pos_rr.txt")
fasta_dict = proc.load_fasta()

pairs = proc.load_pair_list()

# TODO: iterate over all pairs in the list
id, seq = proc.process_pair(pairs[0], fasta_dict)
print(id, seq)

# TODO: adjust the code for GPU and CPU versions
model = AutoModel.from_pretrained('tattabio/gLM2_650M', torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M', trust_remote_code=True)

encodings = tokenizer([seq], return_tensors='pt')

with torch.no_grad():
    embeddings = model(encodings.input_ids, output_hidden_states=True).last_hidden_state

print(embeddings[0])

# TODO: gather embeddings into a dictionary

# TODO: save the dictionary into a PT file