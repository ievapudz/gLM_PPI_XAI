#!/bin/sh

# Testing case of the flow with the batch of embeddings generation

rm -rf tests/outputs/_001/

python3 main.py fit -c tests/configs/001.yaml 

rm -rf tests/outputs/_001/
