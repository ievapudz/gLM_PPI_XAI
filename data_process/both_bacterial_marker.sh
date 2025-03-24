#!/bin/bash

# Taking a subset of filenames (systems) of PINDER that are both from
# bacteria

# Input files
FILENAME_LIST=$1 
BACTERIAL_LIST=$2

# Read UniProt IDs into an associative array for quick lookup
declare -A UNIPROT_IDS
while IFS= read -r ID; do
    UNIPROT_IDS["$ID"]=1
done < "$BACTERIAL_LIST"

# Process filenames and check UniProt IDs
while IFS= read -r FILENAME; do
    # Extract both UniProt IDs using awk
    UNIPROT1=$(echo "$FILENAME" | awk -F'[_-]' '{print $4}')
    echo "1 $UNIPROT1"
    UNIPROT2=$(echo "$FILENAME" | awk -F'[_-]' '{print $9}')
    echo "2 $UNIPROT2"

    # Check if both IDs exist in the reference file
    if [[ -n "${UNIPROT_IDS[$UNIPROT1]}" && -n "${UNIPROT_IDS[$UNIPROT2]}" ]]; then
        echo "$FILENAME,1"
    else
        echo "$FILENAME,0"
    fi
done < "$FILENAME_LIST"
