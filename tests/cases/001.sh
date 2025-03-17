#!/bin/sh

# Testing case of the flow with the generated batches of embeddings

python3 main.py fit -c tests/configs/001.yaml 

rm -rf tests/outputs/001/
