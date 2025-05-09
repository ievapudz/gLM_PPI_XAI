from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from data_process.Processor import Processor
from pathlib import Path
import torch
import os

class UQRDataset(Dataset):
    """
    torch Dataset for upper-right quadrants (of a symmetrical 
    categorical Jacobian or a contact map)
    """

    def __init__(
        self,
        data_file,
        fasta_file,
        uqr_dir,
        num_samples: int = None,
        concat_type = "gLM2"
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

        self.uqrs = self.load_contact_uqrs(uqr_dir)

    def load_contact_uqrs(self, uqr_dir):
        uqrs = {}
        
        for index, row in self.data.iterrows():
            concat_id = row['protein1'] + '-' + row['protein2']
            concat_id = concat_id.translate(str.maketrans({'_': '-', '-': '_'}))

            if(row['label']):
                uqr_path = f"{uqr_dir}/{concat_id}.pt"
                uqrs[concat_id] = torch.load(uqr_path) if(os.path.exists(uqr_path)) else torch.empty(512, 512).fill_(float('nan')) 
            else:
                uqrs[concat_id] = torch.zeros(512, 512)
        
        return uqrs

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
        row['uqr'] = self.uqrs[pair_id]
        return row

class UQRDataModule(LightningDataModule):
    """
    LightningDataModule for SequencePairDataset
    """

    def __init__(
        self,
        fasta_file,
        data_folder,
        uqr_folder,
        batch_size: int,
        positive_only: bool = False,
        num_workers: int = 1,
        num_samples: int = None,
        concat_type: str = "gLM2"
    ):
        super().__init__()
        self.fasta_file = Path(fasta_file)
        self.data_folder = Path(data_folder)
        self.uqr_folder = Path(uqr_folder)
        self.train_file = self.data_folder / "train.txt"
        self.val_file = self.data_folder / "validate.txt"
        self.test_file = self.data_folder / "test.txt"
        self.batch_size = batch_size
        self.positive_only = positive_only
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.concat_type = concat_type

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = UQRDataset(
                self.train_file,
                self.fasta_file,
                self.uqr_folder,
                self.num_samples,
                self.concat_type
            )
            self.val_dataset = UQRDataset(
                self.val_file,
                self.fasta_file,
                self.uqr_folder,
                self.num_samples,
                self.concat_type
            )

        if stage == "test":
            self.test_dataset = UQRDataset(
                self.test_file,
                self.fasta_file,
                self.uqr_folder,
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
