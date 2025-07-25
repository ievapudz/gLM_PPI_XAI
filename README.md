# gLMs + PPI + XAI

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

## Collecting a bacterial PINDER subset

Retrieving UniProt IDs:

```
srun cat ~/projects/gLM/data/PINDER/filenames_[split]_5_1024_400.txt | \
    awk -F'[_-]' '{print $4, $9}' | \
    awk '{if($1 == $2) print $1; else print $1"\n"$2}' \
    > ~/projects/gLM/data/PINDER/uniprotids_[split]_5_1024_400.txt 
```

Retrieving the subset of UniProt IDs that come from bacterial organisms.
Run on transfer node:

```
./data_process/uniprot_api_taxonomy_filter.sh \
    data/PINDER/uniprotids_train_5_1024_400.txt \
    data/PINDER/uniprotids_train_5_1024_400_bacterial.txt

./data_process/uniprot_api_taxonomy_filter.sh \
    data/PINDER/uniprotids_val_5_1024_400.txt \
    data/PINDER/uniprotids_val_5_1024_400_bacterial.txt

./data_process/uniprot_api_taxonomy_filter.sh \
    data/PINDER/uniprotids_test_5_1024_400.txt \
    data/PINDER/uniprotids_test_5_1024_400_bacterial.txt
```

Retrieving the subset of filenames that have only bacterial systems:

```
srun ./data_process/both_bacterial_marker.sh \
    data/PINDER/filenames_train_5_1024_400.txt \
    data/PINDER/uniprotids_train_5_1024_400_bacterial.txt | \
    tr ',' ' ' | awk '{if($2 == 1) print $1}' \
    > data/PINDER/filenames_train_5_1024_400_bacterial.txt 

srun ./data_process/both_bacterial_marker.sh \
    data/PINDER/filenames_val_5_1024_400.txt \
    data/PINDER/uniprotids_val_5_1024_400_bacterial.txt | \
    tr ',' ' ' | awk '{if($2 == 1) print $1}' \
    > data/PINDER/filenames_val_5_1024_400_bacterial.txt 

srun ./data_process/both_bacterial_marker.sh \
    data/PINDER/filenames_test_5_1024_400.txt \
    data/PINDER/uniprotids_test_5_1024_400_bacterial.txt | \
    tr ',' ' ' | awk '{if($2 == 1) print $1}' \
    > data/PINDER/filenames_test_5_1024_400_bacterial.txt 
```


