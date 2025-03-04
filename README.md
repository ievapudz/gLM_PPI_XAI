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

## Access the MLFlow board (deprecated?)

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host localhost --port 8080
```

## Running of the predictions

Example for the entropy predictions:

```
./sbatch_gpu_test.sh Bernett2022-entropy-gpu configs/Bernett2022_1k_entropy.yaml 
```

## Visualisations

Categorical Jacobians:

```
srun --mem-per-cpu=8GB --cpus-per-task=1 --reservation=schwede bash batch_visualise_CJ.sh data/Bernett2022/false_positive_sample.lst 
```

Line detection from entropy matrices:

```
python3 data_process/detect_lines.py -i P23763_P18847 -l data/Bernett2022/lengths.csv
```

## Computation of entropy

For single proteins:

```
srun --mem-per-cpu=8GB --cpus-per-task=1 --reservation=schwede python3 scoring/compute_entropy.py -i data/Bernett2022/fp_proteins.lst
```

For the complexes:

```
srun --mem-per-cpu=8GB --cpus-per-task=1 --reservation=schwede python3 scoring/compute_entropy.py -i data/Bernett2022/false_positive_sample.lst -c
```

## Notes

1. For PyTorch-Lightning to accept MLflow logger, we need to override the setup method of SaveConfigCallback class as given [here](https://github.com/Lightning-AI/pytorch-lightning/discussions/14047).

