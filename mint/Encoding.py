from Bio import SeqIO
import sys
from baselines import init_model
import pandas as pd

class Encoding():
    def __init__(self, emb_type, model_name, devices, batch_size, max_seq_length, pool_type):
        self.emb_type = emb_type
        if(emb_type == "joint_seq_sep_emb"):
            self.model = init_model(model_name, devices, batch_size, max_seq_length, pool_type="sep_"+pool_type)
        else:
            self.model = init_model(model_name, devices, batch_size, max_seq_length, pool_type)

    def load_fasta(fasta_file, emb_type):
        # fasta_file - [STR] of path to a FASTA file with protein sequences
        if(emb_type == "single_seq"):
            sequences = []
            for seq_record in SeqIO.parse(fasta_file, "fasta"):
                sequences.append(str(seq_record.seq))
        else:
            sequences = {}
            for seq_record in SeqIO.parse(fasta_file, "fasta"):
                sequences[str(seq_record.id)] = str(seq_record.seq)
        return sequences

    def pair_sequences(sequences, pair_list):
        # sequences - [DICT] with FASTA headers keys and sequences values
        # pair_list - [STR] of path to a TSV file with paired sequence headers

        seqs1 = []
        seqs2 = []
        pairs = pd.read_csv(pair_list, sep="\t")
        for index, pair in pairs.iterrows():
            seqs1.append(sequences[pair["protein1"]])
            seqs2.append(sequences[pair["protein2"]])
            
        return seqs1, seqs2

    def join_sequences(seqs1, seqs2):
        # seqs1 - [LIST] of [STR] protein sequences
        # seqs2 - [LIST] of [STR] protein sequences
        return [s1 + s2 for s1, s2 in zip(seqs1, seqs2)], [len(s1) for s1 in seqs1]
        
    def single_seq(self, sequences):
        # sequences - [LIST] of [STR] protein sequences
        # Single sequence mode
        embeddings = self.model.encode(sequences, len1=None)
        return embeddings
        
    def joint_seq_joint_emb(self, seqs1, seqs2):
        # seqs1 - [LIST] of [STR] protein sequences
        # seqs2 - [LIST] of [STR] protein sequences
        
        # Joint sequences and jointly pooled per-residue embeddings
        sequences, _ = Encoding.join_sequences(seqs1, seqs2)
        embeddings = self.model.encode(sequences)
        return embeddings
        
    def joint_seq_sep_emb(self, seqs1, seqs2):
        # seqs1 - [LIST] of [STR] protein sequences
        # seqs2 - [LIST] of [STR] protein sequences
        
        # Joint sequences and separately pooled per-residue embeddings

        # DEBUG: because it does not work as it should
        
        sequences, len_seq1 = Encoding.join_sequences(seqs1, seqs2)
        embeddings = self.model.encode(sequences, len1=len_seq1)
        print(embeddings.shape)
        return embeddings
        
    def sep_seq_sep_emb(self, seqs1, seqs2):
        # seqs1 - [LIST] of [STR] protein sequences
        # seqs2 - [LIST] of [STR] protein sequences
        
        # Separate sequences and separately pooled per-residue embeddings
        embeddings = self.model.encode_two(seqs1, seqs2, how="cat")
        return embeddings

    def get_type(self):
        return self.emb_type
        
    def run(self, fasta, pair_list):
        # fasta - [STR] of path to a FASTA file with protein sequences
        # pair_list - [STR] of path to a TSV file with paired sequence headers
        sequences = Encoding.load_fasta(fasta, self.emb_type)
        
        if(self.emb_type == "single_seq"):
            return self.single_seq(sequences)
        
        seqs1, seqs2 = Encoding.pair_sequences(sequences, pair_list)
        
        if(self.emb_type == "joint_seq_joint_emb"):
            return self.joint_seq_joint_emb(seqs1, seqs2)
        elif(self.emb_type == "joint_seq_sep_emb"):
            return self.joint_seq_sep_emb(seqs1, seqs2)
        elif(self.emb_type == "sep_seq_sep_emb"):
            return self.sep_seq_sep_emb(seqs1, seqs2)
        else:
            print("Invalid embeddings type.", file=sys.stderr)
            return 1
        
        