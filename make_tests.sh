#!/bin/sh

CASE=$1

bash tests/cases/${CASE}.sh 2>&1 | \
    sed 's/at \(.*\) line [0-9][0-9]*\./at \1 line 999\./' | \
    sed 's/\x1b/\n/g' | sed 's/\r/\n/g' | \
    sed 's/\(.*\) params/999 params/' | \
    sed 's/Epoch \(.*\)/Epoch 999/' | \
    sed 's/Training\(:\|\s\)\(.*\)/Training/' | \
    sed 's/Validation\(:\|\s\)\(.*\)/Validation/' | \
    sed 's/Sanity \(.*\)/Sanity/' | \
    sed 's/wandb sync \(.*\)/wandb sync -/' > tests/outputs/${CASE}.out
