from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
from data_process.Processor import Processor
from data_process.EmbeddingsGenerator import EmbeddingsGenerator
from pathlib import Path
import os
import torch

class SequencePairDataset(Dataset):
    """
    torch Dataset for sequence pairs
    """

    def __init__(
        self,
        fasta_file,
        data_file,
        num_samples: int = None,
    ):
        self.processor = Processor(fasta_file, data_file)
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


class SequencePairDataModule(LightningDataModule):
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
        emb_dir: str = None,
        batch_dir: str = None
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
        self.emb_dir = emb_dir
        self.batch_dir = batch_dir

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SequencePairDataset(
                self.fasta_file,
                self.train_file,
                self.num_samples,
            )
            self.val_dataset = SequencePairDataset(
                self.fasta_file,
                self.val_file,
                self.num_samples,
            )

        if stage == "test":
            self.test_dataset = SequencePairDataset(
                self.fasta_file,
                self.test_file,
                self.num_samples,
            )

    def exists_batch(self, stage_prefix):
        if(not os.path.exists(self.batch_dir)): return False
        return any(f.startswith(stage_prefix) for f in os.listdir(self.batch_dir))

    def exists_pt(self):
        if(not os.path.exists(self.emb_dir)): return False
        return any(file.endswith(".pt") for file in os.listdir(self.emb_dir))

    def generate_embeddings(self, loader):
        emb_generator = EmbeddingsGenerator(self.emb_dir)
        for batch in loader:
            emb_generator(batch)

    def save_batch_files(self, loader, stage="train"):
        os.makedirs(self.batch_dir, exist_ok=True)

        for batch_idx, batch in enumerate(loader):
            batch_dict = {
                "concat_id": batch["concat_id"],
                "embeddings": None
            }
            batch_path = os.path.join(self.batch_dir, f"{stage}_batch_{batch_idx}.pt")
            for pair_idx, pair in enumerate(batch["concat_id"]):
                if(pair_idx):
                    emb = torch.load(f"{self.emb_dir}/{pair}.pt")
                    batch_dict["embeddings"] = torch.cat((batch_dict["embeddings"], emb))
                else:
                    batch_dict["embeddings"] = torch.load(f"{self.emb_dir}/{pair}.pt")
            torch.save(batch_dict, batch_path)
            print(f'Saved: {batch_path} [Batch size: {len(batch["concat_id"])}]')

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if(not self.exists_batch(stage_prefix="train")):
            if(not self.exists_pt()): self.generate_embeddings(loader)
            self.save_batch_files(loader, stage="train")

        return loader 
    
    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if(not self.exists_batch(stage_prefix="validate")):
            if(not self.exists_pt()): self.generate_embeddings(loader)
            self.save_batch_files(loader, stage="validate")
        return loader 

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if(not self.exists_batch(stage_prefix="test")):
            if(not self.exists_pt()): self.generate_embeddings(loader)
            self.save_batch_files(loader, stage="test")
        return loader

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
