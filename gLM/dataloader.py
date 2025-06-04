from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
from data_process.Processor import Processor
from data_process.EmbeddingsGenerator import EmbeddingsGenerator
from pathlib import Path
import os
import torch
from sklearn.model_selection import KFold

class EmbeddingsDataset(Dataset):
    """
    torch Dataset for sequence pairs represented in embeddings
    """

    def __init__(
        self,
        data_file,
        pt_file,
        split: str, 
        num_samples: int = None,
    ):

        self.processor = Processor(None, data_file, None)
        self.data = self.processor.load_pair_list(sep="\t")

        if num_samples is not None:
            self.data = self.data.head(num_samples)

        print(
            "Num positive pairs:",
            len(self.data[self.data["label"] == 1]),
            "\nNum negative pairs:",
            len(self.data[self.data["label"] == 0]),
        )
        self.embeddings = torch.load(pt_file)
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        item is a concatenated sequence of the pair
        """
        row = self.data.iloc[idx].to_dict()
        if(self.split == "train" or self.split == "validate"):
            row['concat_id'] = idx
        else:
            row['concat_id'] = f"{row['protein1']}_{row['protein2']}"
        row['embeddings'] = self.embeddings[idx]
        return row


class EmbeddingsDataModule(LightningDataModule):
    """
    LightningDataModule for SequencePairDataset
    """

    def __init__(
        self,
        data_folder,
        pt_folder,
        batch_size: int,
        positive_only: bool = False,
        num_workers: int = 1,
        num_samples: int = None,
    ):
        super().__init__()
        self.data_folder = Path(data_folder)
        self.pt_folder = Path(pt_folder)
        self.train_file = self.data_folder / "train.txt"
        self.val_file = self.data_folder / "validate.txt"
        self.test_file = self.data_folder / "test.txt"
        self.train_pt_file = self.pt_folder / "train.pt"
        self.val_pt_file = self.pt_folder / "val.pt"
        self.test_pt_file = self.pt_folder / "test.pt"
        self.batch_size = batch_size
        self.positive_only = positive_only
        self.num_workers = num_workers
        self.num_samples = num_samples

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = EmbeddingsDataset(
                self.train_file,
                self.train_pt_file,
                "train",
                self.num_samples,
            )
            self.val_dataset = EmbeddingsDataset(
                self.val_file,
                self.val_pt_file,
                "validate",
                self.num_samples,
            )

        if stage == "test":
            self.test_dataset = EmbeddingsDataset(
                self.test_file,
                self.test_pt_file,
                "test",
                self.num_samples
            )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader 
    
    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader 

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

class SequencePairDataset(Dataset):
    """
    torch Dataset for sequence pairs
    """

    def __init__(
        self,
        fasta_file,
        data_file,
        num_samples: int = None,
        concat_type: str = "gLM2"
    ):
        self.processor = Processor(fasta_file, data_file, concat_type)
        self.data = self.processor.load_pair_list()

        if num_samples is not None:
            self.data = self.data.head(num_samples)

        print(
            "Num positive pairs:",
            len(self.data[self.data["label"] == 1]),
            "\nNum negative pairs:",
            len(self.data[self.data["label"] == 0]),
        )
        self.fasta_file = fasta_file
        self.fasta_dict = self.processor.load_fasta()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        item is a concatenated sequence of the pair
        """
        row = self.data.iloc[idx].to_dict()
        pair_id, seq, len1, len2 = self.processor.process_pair(
            [row['protein1'], row['protein2']], self.fasta_dict, aa_only=True)
        row['concat_id'] = pair_id
        row['sequence'] = seq
        row['length1'] = len1+1
        row['length2'] = len2+1
        return row

class SequencePairCJDataModule(LightningDataModule):
    """
    LightningDataModule for SequencePairDataset
    """

    def __init__(
        self,
        fasta_file,
        data_folder,
        batch_size: int,
        positive_only: bool = False,
        num_workers: int = 1,
        num_samples: int = None,
        concat_type: str = "gLM2",
        kfolds: int = 1,
        kfold_idx: int = 0,
        seed: int = 42
    ):
        super().__init__()
        self.fasta_file = Path(fasta_file)
        self.data_folder = Path(data_folder)
        self.train_file = self.data_folder / "train.txt"
        self.val_file = self.data_folder / "validate.txt"
        self.test_file = self.data_folder / "test.txt"
        self.batch_size = batch_size
        self.positive_only = positive_only
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.concat_type = concat_type
        self.kfolds = kfolds
        self.kfold_idx = kfold_idx
        self.seed = seed

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if(self.kfolds > 1):
                full_dataset = SequencePairDataset(
                    self.fasta_file,
                    self.train_file,
                    self.num_samples,
                    self.concat_type
                )
                kf = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)
                all_splits = [k for k in kf.split(full_dataset)]
                train_indexes, val_indexes = all_splits[self.kfold_idx]
                train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
                self.train_dataset, self.val_dataset = full_dataset[train_indexes], dataset_full[val_indexes]

            else:
                self.train_dataset = SequencePairDataset(
                    self.fasta_file,
                    self.train_file,
                    self.num_samples,
                    self.concat_type
                )
                self.val_dataset = SequencePairDataset(
                    self.fasta_file,
                    self.val_file,
                    self.num_samples,
                    self.concat_type
                )

        if stage == "test":
            self.test_dataset = SequencePairDataset(
                self.fasta_file,
                self.test_file,
                self.num_samples,
                self.concat_type
            )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

