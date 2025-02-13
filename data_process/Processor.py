from Bio import SeqIO

class Processor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_fasta(self):
        headers = []
        seqs = []
        with open(self.file_path, "r") as fh:
            for record in SeqIO.parse(fh, "fasta"):
                headers.append(record.id)
                seqs.append(str(record.seq))
        fasta = dict(zip(headers, seqs))
        return fasta
    
    def process_pair(self, prot1, prot2, fasta_dict, aa_only=True):
        pair_id = f"{prot1.replace('-', '_')}-{prot2.replace('-', '_')}"
        if(aa_only): concat_seq = f"<+>{fasta_dict[prot1]}<+>{fasta_dict[prot2]}"
        return (pair_id, concat_seq)

proc = Processor("../data/Bernett2022/human_swissprot_oneliner.fasta")
fasta_dict = proc.load_fasta()
id, seq = proc.process_pair("O15121", "P54886", fasta_dict)
print(id, seq)
