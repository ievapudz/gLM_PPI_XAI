#!/usr/bin/env python3

from optparse import OptionParser
import json
from pathlib import Path
from Bio import SeqIO
import gzip

FAKESEQ="FAKESEQUENCEFAKESEQUENCEFAKESEQUENCEFAKESEQUENCE"

parser = OptionParser()

parser.add_option("--genomic-json", "-g", dest="genomic_json",
	default=None, help="path to the input JSON file with genomic context info.")

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

def main():
    with open(genomic_json, "r") as f:
        data = json.load(f)

    i = 0
    for uniprot_id, content in data.items():
        if(i > 5): break
        print(uniprot_id)
        mid_idx = int(len(content["flanking_genes"]["ncbi_codes"])/2)

        # Retrieving aminoacid sequences
        upstream_flanking_seqs = []
        for j in range(context_size, 0, -1):
            flanking_seq = content["flanking_genes"]["sequences"][mid_idx-j]
            if(not flanking_seq or flanking_seq == FAKESEQ):
                seq = search_faa_file(content, content["flanking_genes"]["ncbi_codes"][mid_idx-j])
                upstream_flanking_seqs.append(seq)
            else: 
                upstream_flanking_seqs.append(flanking_seq)
                
        downstream_flanking_seqs = []
        for j in range(1, context_size+1):
            flanking_seq = content["flanking_genes"]["sequences"][mid_idx+j]
            if(flanking_seq or flanking_seq == FAKESEQ):
                seq = search_faa_file(content, content["flanking_genes"]["ncbi_codes"][mid_idx+j])
                downstream_flanking_seqs.append(seq)
            else:
                downstream_flanking_seqs.append(flanking_seq)

        print(upstream_flanking_seqs)
        print(downstream_flanking_seqs)
        
        contextualised_seq = ""
        for s in upstream_flanking_seqs:
            contextualised_seq += f"<+>{s}"

        mid_seq = content["flanking_genes"]["sequences"][mid_idx]
        
        if(mid_seq or mid_seq == FAKESEQ):
            seq = search_faa_file(content, content['flanking_genes']['ncbi_codes'][mid_idx])
            contextualised_seq += f"<+>{seq}"
        else:
            contextualised_seq += f"<+>{content['flanking_genes']['sequences'][mid_idx]}"

        for s in downstream_flanking_seqs:
            contextualised_seq += f"<+>{s}"

        #print(f">{uniprot_id}")
        #print(contextualised_seq)
    
        i += 1 

if __name__ == "__main__":
    main()