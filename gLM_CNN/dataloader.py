from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from data_process.Processor import Processor
from pathlib import Path
import torch
import os
import torch.multiprocessing
from sklearn.model_selection import KFold

torch.multiprocessing.set_sharing_strategy('file_system')

class URQDataset(Dataset):
    """
    torch Dataset for upper-right quadrants (of a symmetrical 
    categorical Jacobian or a contact map)
    """

    def __init__(
        self,
        data_file,
        fasta_file,
        urq_dir,
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

        self.urqs = self.load_contact_urqs(urq_dir)

    def load_contact_urqs(self, urq_dir):
        urqs = {}
        
        for index, row in self.data.iterrows():
            concat_id = row['protein1'] + '-' + row['protein2']
            concat_id = concat_id.translate(str.maketrans({'_': '-', '-': '_'}))

            if(row['label']):
                urq_path = f"{urq_dir}/{concat_id}.pt"
                urqs[concat_id] = torch.load(urq_path) if(os.path.exists(urq_path)) else torch.empty(512, 512).fill_(float('nan')) 
            else:
                urqs[concat_id] = torch.zeros(512, 512)
        
        return urqs

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
        row['urq'] = self.urqs[pair_id]
        return row

class URQDataModule(LightningDataModule):
    """
    LightningDataModule for URQDataset
    """

    def __init__(
        self,
        fasta_file,
        data_folder,
        urq_folder,
        tensor_dir: str,
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
        self.urq_folder = Path(urq_folder)
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
    
        self.tensor_dir = tensor_dir

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if(self.kfolds > 1):
                full_dataset = URQDataset(
                    self.train_file,
                    self.fasta_file,
                    self.urq_folder,
                    self.num_samples,
                    self.concat_type
                )
                kf = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)
                all_splits = [k for k in kf.split(full_dataset)]
                train_indexes, val_indexes = all_splits[self.kfold_idx]
                train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
                self.train_dataset, self.val_dataset = full_dataset[train_indexes], dataset_full[val_indexes]

            else:
                self.train_dataset = URQDataset(
                    self.train_file,
                    self.fasta_file,
                    self.urq_folder,
                    self.num_samples,
                    self.concat_type,
                )
                self.val_dataset = URQDataset(
                    self.val_file,
                    self.fasta_file,
                    self.urq_folder,
                    self.num_samples,
                    self.concat_type,
                )

        if stage == "test":
            self.test_dataset = URQDataset(
                self.test_file,
                self.fasta_file,
                self.urq_folder,
                self.num_samples,
                self.concat_type,
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
