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

def extract_gene_info(uniprot_id, gff_file, product_accession, content, idx):
    relative_start = content["flanking_genes"]["relative_starts"][idx]
    relative_end = content["flanking_genes"]["relative_ends"][idx]
    direction = content["flanking_genes"]["directions"][idx]
    
    with open(gff_file) as f:
        for line in f:
            if line.startswith("#") or product_accession not in line:
                continue
            fields = line.strip().split('\t')
            contig_id = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            return {
                "uniprot_id": uniprot_id,
                "contig_id": contig_id,
                "start": start,
                "end": end,
                "strand": strand,
                "relative_start": relative_start,
                "relative_end": relative_end,
                "direction": direction
            }
    return None

def get_region_boundaries(genes_info, goi_idx):
    # TODO: LIKELY NOT NEEDED AT ALL
    
    # The determination of the genomic coordinates
    if(None in genes_info): return (0, 0)

    # Given coordinates
    if(genes_info[goi_idx]["strand"] == "+"):
    print(genes_info[goi_idx]["strand"], genes_info[0]["start"], genes_info[-1]["end"])
    print(genes_info)
    
    assemblies = []
    for gene in genes_info:
        assemblies.append(gene["contig_id"])

    if(len(set(assemblies)) != 1): 
        print(f'Genomic context made of different contigs for {genes_info[goi_idx]["uniprot_id"]}')
    
    return (genes_info[0]["start"], genes_info[-1]["end"])

    
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

    return str(whole_region_seq), str(upstream_region_seq), str(downstream_region_seq) 

def get_cds(content, goi_idx, context_size):
    cds = []
    for i in range(-context_size, context_size+1):
        cds.append(content["flanking_genes"]["sequences"][goi_idx+i])
    return cds

def get_igs(genes_info, fna_file, contig_id)
    igs = []

    for gene in genes_info:
        records = SeqIO.to_dict(SeqIO.parse(fna_file, "fasta"))
        if gene["contig_id"] not in records:
            raise ValueError(f"Contig {contig_id} not found in {fna_file}")

        seq_record = records[contig_id]
        seq_len = len(seq_record.seq)

        # Get whole region to in nucleotides
    
    """
    
    # TODO: this is now adjusted for upstream-only and direction +
    
    for i in range(context_size):
        if(starts[i+1]):
            igs.append(regions[0][ends[i]:starts[i+1]])
        else:
            igs.append(regions[0][ends[i]:])
        
        print(len(regions[0]), ends[i], starts[i+1], starts[i+1]-ends[i], len(igs[-1]))

    goi_length = ends[int(len(ends)/2)] - starts[int(len(ends)/2)]
    for i in range(context_size, context_size*2):
        igs.append(regions[1][ends[i]-goi_length:starts[i+1]-goi_length])
        print(len(regions[1]), ends[i], starts[i+1], starts[i+1]-ends[i], len(igs[-1]))
    """
    return igs

def main():
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

        genes_info = []
        goi_idx = int(len(content["flanking_genes"]["ncbi_codes"])/2)
        
        for j in range(-context_size, context_size+1):
            gene_info = extract_gene_info(uniprot_id, gff_path, content["flanking_genes"]["ncbi_codes"][goi_idx+j], content, goi_idx+j)
            genes_info.append(gene_info)

        goi_idx = int(len(genes_info)/2)

        cds = get_cds(content, goi_idx, context_size)
        igs = get_igs(genes_info, fna_file, contig_id)
        
        # TODO: save sequences with the headers that denote the token idx that ends the upstream genomic context and starts the downstream genomic context. It is needed for further processing of the sequences
        
        
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

