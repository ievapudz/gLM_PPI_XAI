import os
import re
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from optparse import OptionParser
import json

parser = OptionParser()

parser.add_option("--genomic-json", "-g", dest="genomic_json",
	default=None, help="path to the input JSON file with genomic context info.")

parser.add_option("--output-fasta", "-o", dest="output_fasta",
    default=None, help="path to the output mixed-modality FASTA file.")

parser.add_option("--context-size", "-c", dest="context_size",
    default=100, help="number of flanking genes to include.")

parser.add_option("--nt-only", "-n", dest="nt_only",
    default=False, action="store_true", help="option to have genomic context only of nucleotides.")

parser.add_option("--dir", "-d", dest="ncbi_dir",
    default="data/PINDER/eubacteria_5_1024_512_species_heterodimers/genomic/ncbi_complete_bacteria/ncbi_dataset/data/",
    help="path to the directory with fetched genomic assemblies")

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

def get_flanking_gene_info(gff_path, content, idx):
    gene_info = extract_gene_info(gff_path, 
        content["flanking_genes"]["ncbi_codes"][idx]
    )
    relative_start = content["flanking_genes"]["relative_starts"][idx]
    relative_end = content["flanking_genes"]["relative_ends"][idx]

    if(gene_info):
        contig_id, start, end, strand = gene_info
        return contig_id, start, end, strand, relative_start, relative_end
    else:
        return None

def extract_genomic_region(fna_file, contig_id, region_start, region_end, strand, goi_start=None, goi_end=None):
    records = SeqIO.to_dict(SeqIO.parse(fna_file, "fasta"))
    if contig_id not in records:
        raise ValueError(f"Contig {contig_id} not found in {fna_file}")
    
    seq_record = records[contig_id]
    seq_len = len(seq_record.seq)

    # Retrieval of region in nts
    whole_region_seq = seq_record.seq[region_start:region_end].lower()

    # Retrieval of flanking nt regions
    upstream_region_seq = None
    downstream_region_seq = None
    if(goi_start):
        upstream_region_seq = seq_record.seq[region_start:goi_start].lower()
    if(goi_end):
        downstream_region_seq = seq_record.seq[goi_end:region_end].lower()
        
    if strand == '-':
        whole_region_seq = whole_region_seq.reverse_complement()

    # TODO: combine the CDS regions in aa
    # - Get aa sequences from JSON
    # - Get absolute coordinates of each subregion to know where to slice the nt sequence

    return str(whole_region_seq), str(upstream_region_seq), str(downstream_region_seq) 

def main():
    # TODO: get assembly IDs of the product from JSON
    with open(genomic_json, "r") as f:
        data = json.load(f)

    i = 0
    for uniprot_id, content in data.items():
        if(i > 5): break
        assembly_accession = content["assembly_id"][1]
        directory = Path(f"{options.ncbi_dir}/{assembly_accession}")
        gff_path = directory / "genomic.gff"
        fna_path = next(directory.glob("*_genomic.fna"), None)
        if not gff_path.exists() or fna_path is None:
            print(f"Missing files for {assembly}")
            continue

        # Gene of interest
        goi_info = extract_gene_info(gff_path, content["assembly_id"][0])
        goi_idx = int(len(content["flanking_genes"]["ncbi_codes"])/2)
        # Info about the first upstream flanking gene
        first_ufg_info = get_flanking_gene_info(gff_path, content, goi_idx-context_size)
        # Info about the last downstream flanking gene
        last_dfg_info = get_flanking_gene_info(gff_path, content, goi_idx+context_size)
        
        if goi_info is None:
            print(f"Product {content['assembly_id'][0]} not found in {gff_path}")
            continue

        print("***", content["assembly_id"][0], goi_info)
        print(first_ufg_info)
        print(last_dfg_info)

        # The determination of the genomic coordinates
        if(goi_info[3] == "+" and first_ufg_info):
            start_region = goi_info[1] + first_ufg_info[4] - 1
            end_region = goi_info[1] + last_dfg_info[5] - 1
            print("Given coordinates: ", first_ufg_info[1], last_dfg_info[2])
            print("Computed coordinates: ", start_region, end_region)
        elif(goi_info[3] == "-" and first_ufg_info):
            start_region = goi_info[2] - last_dfg_info[5] + 1
            end_region = goi_info[2] - first_ufg_info[4] + 1
            print("Given coordinates: ", last_dfg_info[1], first_ufg_info[2])
            print("Computed coordinates: ", start_region, end_region)

        if(first_ufg_info and last_dfg_info):
            # Now retrieval of the region in nts
            whole_seq, u_seq, d_seq = extract_genomic_region(fna_path, goi_info[0], start_region, end_region, goi_info[3], goi_start=goi_info[1], goi_end=goi_info[2])
            
            print(len(whole_seq), end_region-start_region, len(whole_seq)-len(u_seq)-len(d_seq), goi_info[2]-goi_info[1])

        # TODO: save sequences with the headers that denote the token idx that ends the upstream genomic context and starts the downstream genomic context. It is needed for further processing of the sequences
        
        """
        contig_id, start, end, strand = goi_info
        try:
            context_seq, region_start, region_end = extract_genomic_region(fna_path, contig_id, start, end, strand, context_size)
        except ValueError as e:
            print(e)
            continue

        print(assembly, product, context_seq)
        """
        i += 1           

if __name__ == "__main__":
    main()

