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
mamba install sympy
mamba install scikit-learn

mamba install pytorch_lightning
pip3 install -U 'jsonargparse[signatures]>=4.27.7'
pip3 install mlflow
```

## Download gLM

```
ssh [username]@transfer12.scicore.unibas.ch

git clone https://huggingface.co/tattabio/gLM2_650M
```

## Access the MLFlow board

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host localhost --port 8080
```

## Notes

1. For PyTorch-Lightning to accept MLflow logger, we need to override the setup method of SaveConfigCallback class as given [here](https://github.com/Lightning-AI/pytorch-lightning/discussions/14047).

