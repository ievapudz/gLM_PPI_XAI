#!/bin/sh

# Testing case of the flow with generated batch embeddings' files

python3 main.py fit -c tests/configs/003.yaml 

rm -rf tests/outputs/_003/checkpoints/
