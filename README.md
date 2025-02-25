# gLMs + PPI + XAI

## Log into cluster

```
ssh [username]@login12.scicore.unibas.ch
```

## Set up conda environment

```
conda create -n gLM python=3.11

conda activate gLM

pip3 install lightning torch torch-geometric tensorboard wandb 'jsonargparse[signatures]' transformers mlflow torchvision

mamba install pandas numpy matplotlib einops sympy scikit-learn biopython seaborn
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

## Visualisation of categorical Jacobians

```
srun --mem-per-cpu=8GB --cpus-per-task=1 --reservation=schwede bash batch_visualise_CJ.sh data/Bernett2022/false_positive_sample.lst 
```

## Notes

1. For PyTorch-Lightning to accept MLflow logger, we need to override the setup method of SaveConfigCallback class as given [here](https://github.com/Lightning-AI/pytorch-lightning/discussions/14047).

