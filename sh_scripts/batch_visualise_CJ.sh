#!/bin/bash

# Path to the input file
input_file=$1

# Loop through each line in the input file
while IFS= read -r line; do
    # Pass the value of the line as input to the pipeline command
    echo "Processing: $line"
    # Replace 'your_pipeline_command' with the actual command you want to run
    python3 data_process/visualise_CJ.py -i "$line" -l data/Bernett2022/lengths.csv
done < "$input_file"