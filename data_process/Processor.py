from Bio import SeqIO
import pandas as pnd

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
    
    def load_pair_list(self):
        pairs = pnd.read_csv(self.pair_list_path, sep="\t")
        return pairs
    
    def process_pair(self, pair, fasta_dict, aa_only=True):
        pair = pair.split()
        pair_id = f"{pair[0].replace('_', '-')}_{pair[1].replace('_', '-')}"
        if(aa_only): concat_seq = f"<+>{fasta_dict[pair[0]]}<+>{fasta_dict[pair[1]]}"
        return (pair_id, concat_seq)
