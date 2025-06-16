import os
import re
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--assemblies", "-a", dest="assemblies_file",
	default=None, help="path to the input assemblies accession TXT.")

parser.add_option("--products", "-p", dest="products_file",
	default=None, help="path to the input product accession TXT.")

parser.add_option("--protein-fasta", dest="protein_fasta",
	default=None, help="path to the input protein FASTA file.")

parser.add_option("--output-fasta", "-o", dest="output_fasta",
    default=None, help="path to the output mixed-modality FASTA file.")

parser.add_option("--context-size", "-c", dest="context_size",
    default=100, help="length of the context to add to before and after.")

parser.add_option("--dir", "-d", dest="ncbi_dir",
    default="data/PINDER/eubacteria_5_1024_512_species_heterodimers/genomic/ncbi_complete_bacteria/ncbi_dataset/data/",
    help="path to the directory with fetched genomic assemblies")

(options, args) = parser.parse_args()

# CONFIGURATION
assemblies_file = options.assemblies_file
products_file = options.products_file
protein_fasta = options.protein_fasta
output_fasta = options.output_fasta
context_size = int(options.context_size)  # nt before and after

def load_mapping(assemblies_path, products_path):
    with open(assemblies_path) as a, open(products_path) as p:
        assemblies = [line.strip() for line in a]
        products = [line.strip() for line in p]
    return list(zip(assemblies, products))

def load_proteins(protein_fasta):
    return {record.id: str(record.seq) for record in SeqIO.parse(protein_fasta, "fasta")}

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

def extract_genomic_region(fna_file, contig_id, start, end, strand, context=500):
    records = SeqIO.to_dict(SeqIO.parse(fna_file, "fasta"))
    if contig_id not in records:
        raise ValueError(f"Contig {contig_id} not found in {fna_file}")
    
    seq_record = records[contig_id]
    seq_len = len(seq_record.seq)

    # Adjust context
    region_start = max(0, start - context - 1)
    region_end = min(seq_len, end + context)

    region_seq = seq_record.seq[region_start:region_end].lower()
    if strand == '-':
        region_seq = region_seq.reverse_complement()

    return str(region_seq), region_start + 1, region_end  # adjust for 1-based coords

def main():
    mapping = load_mapping(assemblies_file, products_file)
    protein_seqs = load_proteins(protein_fasta)

    with open(output_fasta, "w") as out_f:
        i = 0
        for assembly, product in mapping:
            if(i > 5): break
            directory = Path(f"{options.ncbi_dir}/{assembly}")
            print(directory)
            gff_path = directory / "genomic.gff"
            fna_path = next(directory.glob("*_genomic.fna"), None)

            if not gff_path.exists() or fna_path is None:
                print(f"Missing files for {assembly}")
                continue

            gene_info = extract_gene_info(gff_path, product)
            if gene_info is None:
                print(f"Product {product} not found in {gff_path}")
                continue

            contig_id, start, end, strand = gene_info
            try:
                context_seq, region_start, region_end = extract_genomic_region(fna_path, contig_id, start, end, strand, context_size)
            except ValueError as e:
                print(e)
                continue

            print(assembly, product, context_seq)
            """
            # Get the protein sequence
            uniprot_id = [k for k, v in protein_seqs.items() if product in k or product == k]
            if not uniprot_id:
                print(f"Protein for {product} not found in FASTA.")
                continue

            aa_seq = protein_seqs[uniprot_id[0]]

            header = f">{uniprot_id[0]}|strand={strand}|region={region_start}-{region_end}|±{context_size}nt"
            out_f.write(f"{header}\n{context_seq}\n{aa_seq}\n")
            """
            i += 1

if __name__ == "__main__":
    main()

