import json
import sys

def extract_assembly_info(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    assembly_info = {}

    for uniprot_id, content in data.items():
        assembly_entry = content.get("assembly_id", [])

        if len(assembly_entry) < 3:
            print(f"Incomplete assembly_id for {uniprot_id}: {assembly_entry}")
            continue

        product_accession = assembly_entry[0]
        assembly_accession = assembly_entry[1]
        assembly_url = assembly_entry[2]

        assembly_info[uniprot_id] = {
            "product_accession": product_accession,
            "assembly_accession": assembly_accession,
            "ftp_link": assembly_url
        }

    return assembly_info

# Example usage
json_file = sys.argv[1]
out_dir = sys.argv[2]
assembly_data = extract_assembly_info(json_file)

f = open(f"{out_dir}/uniprot_ids.txt", "w")
for uniprot_id, info in assembly_data.items():
    print(uniprot_id, file=f)
f.close()

f = open(f"{out_dir}/product_accessions.txt", "w")
for uniprot_id, info in assembly_data.items():
    print(info['product_accession'], file=f)
f.close()

f = open(f"{out_dir}/assembly_accessions.txt", "w")
for uniprot_id, info in assembly_data.items():
    print(info['assembly_accession'], file=f)
f.close()

f = open(f"{out_dir}/assembly_ftp_links.txt", "w")
for uniprot_id, info in assembly_data.items():
    print(info['ftp_link'], file=f)
f.close()
