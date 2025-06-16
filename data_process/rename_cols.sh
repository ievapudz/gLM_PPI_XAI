#!/bin/bash

LOGS_DIR="../logs/CategoricalJacobianCNN"

# Navigate to your logs directory
cd $LOGS_DIR || exit

# Find all metrics.csv files and process them
find . -type f -path "*/Softmax_CV_I_*/gLM2/*/metrics.csv" | while read -r file; do
    echo "Fixing $file"
    
    # Read the header line
    header=$(head -n 1 "$file")
    
    #fixed_header=$(echo "$header" | sed 's/\btp\b/tn/1')
    fixed_header="epoch,mcc,pr_auc,roc_auc,tn,fp,fn,tp"
    
    # Replace the header in the file
    echo "$fixed_header" > $file".corrected"
    tail -n +2 "$file" >> $file".corrected"
    mv $file".corrected" $file
done

