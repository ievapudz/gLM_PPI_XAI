from Bio import SeqIO
import pandas as pd

class Processor:
    def __init__(self, fasta_path, pair_list_path):
        self.fasta_path = fasta_path
        self.pair_list_path = pair_list_path

    def load_fasta(self):
        headers = []
        seqs = []
        with open(self.fasta_path, "r") as fh:
            for record in SeqIO.parse(fh, "fasta"):
                headers.append(record.id)
                seqs.append(str(record.seq))
        fasta = dict(zip(headers, seqs))
        return fasta
    
    def load_pair_list(self,):
        pairs = pd.read_csv(self.pair_list_path, sep="\t")
        return pairs
    
    def process_pair(self, pair, fasta_dict, aa_only=True, ready_pair_ids=False):
        if(ready_pair_ids):
            pair_id = pair
            pair = pair.split('_')
        else:
            pair_id = f"{str(pair[0]).replace('_', '-')}_{str(pair[1]).replace('_', '-')}"
        if(aa_only): concat_seq = f"<+>{fasta_dict[pair[0]]}<+>{fasta_dict[pair[1]]}"
        len1 = len(fasta_dict[pair[0]])
        len2 = len(fasta_dict[pair[1]])
        return (pair_id, concat_seq, len1, len2)
