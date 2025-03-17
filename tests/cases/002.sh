#!/bin/sh

# Testing case of the flow with the generated embeddings (but not batches)

python3 main.py fit -c tests/configs/002.yaml 

rm -rf tests/outputs/_002/batches/ tests/outputs/_002/checkpoints/
