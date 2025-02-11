# gLMs + PPI + XAI

## Log into cluster

```
ssh [username]@login12.scicore.unibas.ch
```

## Set up conda environment

```
conda create -n gLM python=3.9

conda activate gLM

mamba install pandas
mamba install numpy
mamba install matplotlib

pip3 install torch torchvision torchaudio

mamba install transformers
mamba install einops
```

## Download gLM

```
ssh [username]@transfer12.scicore.unibas.ch

git clone https://huggingface.co/tattabio/gLM2_650M
```

