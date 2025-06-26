#!/usr/bin/env python3

from optparse import OptionParser
import json
from pathlib import Path
import gzip
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

FAKESEQ="FAKESEQUENCEFAKESEQUENCEFAKESEQUENCEFAKESEQUENCE"

parser = OptionParser()

parser.add_option("--genomic-json", "-g", dest="genomic_json",
	default=None, help="path to the input JSON file with genomic context info.")

parser.add_option("--pair-list", "-p", dest="pair_list",
	default=None, help="path to the input list of complexes' ID pairs.")

parser.add_option("--output-fasta", "-o", dest="output_fasta",
    default=None, help="path to the output mixed-modality FASTA file.")

parser.add_option("--context-size", "-c", dest="context_size",
    default=100, help="number of flanking genes to include.")

parser.add_option("--dir", "-d", dest="ncbi_dir",
    default="data/PINDER/eubacteria_5_1024_512_species_heterodimers/genomic/ncbi_complete_bacteria/ncbi_dataset/data/",
    help="path to the directory with fetched genomic assemblies.")

parser.add_option("--database", dest="database",
    default="/scicore/home/schwede/GROUP/gcsnap_db/refseq/data/",
    help="path to the database.")

(options, args) = parser.parse_args()

# CONFIGURATION
genomic_json = options.genomic_json
output_fasta = options.output_fasta
context_size = int(options.context_size)  # num. genes before and after

def read_pair_list(list_path):
    pairs = pd.read_csv(list_path, sep='\t', header=0)
    uniprot_id_pairs = []
    for i in range(len(pairs)):
        uniprot_id_pairs.append((pairs.loc[i, 'protein1'].split('_')[-1], pairs.loc[i, 'protein2'].split('_')[-1]))
    return uniprot_id_pairs

def extract_gene_info(gff_file, product_accession):
    with open(gff_file) as f:
        for line in f:
            if line.startswith("#") or product_accession not in line:
                continue
            fields = line.strip().split('\t')
            contig_id = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            return contig_id, start, end, strand
    return None

def search_faa_file(content, ncbi_code):
    assembly_accession = content["assembly_id"][1]

    db_directory = Path(f"{options.database}")
    faa_path = next(db_directory.glob(f"{assembly_accession}_*.faa.gz"), None)
    if faa_path is None:
        print(f"Missing files for {assembly_accession}")
        return ""

    with gzip.open(faa_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if(ncbi_code in record.id):
                return str(record.seq)
                
    return ""

def get_aa_sequence(content, idx):
    seq = content["flanking_genes"]["sequences"][idx]
    if(not seq or seq == FAKESEQ):
        seq = search_faa_file(content, content["flanking_genes"]["ncbi_codes"][idx])
    return seq

def get_flanking_seqs(uniprot_id, content, mid_idx):

    upstream_flanking_seqs = []
    for j in range(context_size, 0, -1):
        flanking_seq = get_aa_sequence(content, mid_idx-j)
        upstream_flanking_seqs.append(flanking_seq)
            
    downstream_flanking_seqs = []
    for j in range(1, context_size+1):
        flanking_seq = get_aa_sequence(content, mid_idx+j)
        downstream_flanking_seqs.append(flanking_seq)
    
    return upstream_flanking_seqs, downstream_flanking_seqs

def get_contextualised_sequence(mid_seq, upstream_flanking_seqs=None, downstream_flanking_seqs=None):
    contextualised_seq = ""
    if(upstream_flanking_seqs):
        for s in upstream_flanking_seqs:
            contextualised_seq += f"<+>{s}"

    contextualised_seq += f"<+>{mid_seq}"
    if(downstream_flanking_seqs):
        for s in downstream_flanking_seqs:
            contextualised_seq += f"<+>{s}"

    return contextualised_seq

def main():
    with open(genomic_json, "r") as f:
        data = json.load(f)

    # TODO: read the pair list and get their contexts simultaneously
    uniprot_id_pairs = read_pair_list(options.pair_list)

    both_in_json = 0
    fasta = []
    for pair in uniprot_id_pairs:
        if(pair[0] in data.keys() and pair[1] in data.keys()):
            both_in_json += 1

            # Get upstream flanking seqs. of prot1
            content = data[pair[0]]
            mid_idx = int(len(content["flanking_genes"]["ncbi_codes"])/2)
            upstream_flanking_seqs, _ = get_flanking_seqs(pair[0], content, mid_idx)
            prot1_seq = get_aa_sequence(content, mid_idx)
            context_prot1_seq = get_contextualised_sequence(prot1_seq, 
                upstream_flanking_seqs, downstream_flanking_seqs=None
            )

            # Get downstream flanking seqs. of prot2
            content = data[pair[1]]
            mid_idx = int(len(content["flanking_genes"]["ncbi_codes"])/2)
            _, downstream_flanking_seqs = get_flanking_seqs(pair[1], content, mid_idx)
            prot2_seq = get_aa_sequence(content, mid_idx)
            context_prot1_seq = get_contextualised_sequence(prot2_seq, 
                None, downstream_flanking_seqs
            )

            # TODO: write FASTA manually!!!! Because SeqRecord omits tokens
            fasta.append(SeqRecord(Seq(prot1_seq), id=pair[0], description=""))
            fasta.append(SeqRecord(Seq(prot2_seq), id=pair[1], description=""))
            
    print("Pairs that have both uniprot_ids in json: ", both_in_json)

    with open(output_fasta, "w") as output_handle:
        SeqIO.write(fasta, output_handle, "fasta-2line")

if __name__ == "__main__":
    main()