#!/bin/bash

# Filtering of UniProt IDs that contain bacterial entry

# Define the input file containing UniProt IDs (one per line)
INPUT_FILE=$1
OUTPUT_FILE=$2

# Constants
TAX_ID=2
BATCH_SIZE=100
TMP_FILE="batch_ids.txt"

# Clear output file before appending results
> "$OUTPUT_FILE"

# Function to send a request with up to 100 IDs
send_request() {
    local IDS_STRING
    IDS_STRING=$(head -c -1 $TMP_FILE | tr '\n' '*' | sed 's/*/+OR+/g') 
    if [[ -n "$IDS_STRING" ]]; then
        echo "Querying batch: $IDS_STRING"
        RESPONSE=$(curl -s -H "Accept: text/plain; format=list" \
            "https://rest.uniprot.org/uniprotkb/search?query=$IDS_STRING+AND+taxonomy_id:$TAX_ID")
        echo "$RESPONSE" >> "$OUTPUT_FILE"
    fi
}

# Read file line by line and process in chunks of BATCH_SIZE
COUNT=0
> "$TMP_FILE"
while IFS= read -r ID; do
    [[ "$ID" == "UNDEFINED" || -z "$ID" ]] && continue  # Skip 'UNDEFINED' and empty lines

    echo "$ID" >> "$TMP_FILE"
    ((COUNT++))

    # If we reach the batch size, send the request
    if (( COUNT == BATCH_SIZE )); then
        send_request
        > "$TMP_FILE"  # Clear temp file for next batch
        COUNT=0
    fi
done < "$INPUT_FILE"

# Send any remaining IDs
if [[ -s "$TMP_FILE" ]]; then
    send_request
fi

# Preserve only unique entries
sort -u "$OUTPUT_FILE" > "$TMP_FILE"
mv "$TMP_FILE" "$OUTPUT_FILE"

# Cleanup
rm -f "$TMP_FILE"

echo "Results saved to $OUTPUT_FILE"

