from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from data_process.Processor import Processor
from pathlib import Path
import torch
import os
import torch.multiprocessing

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

class URQBatchDataset(Dataset):
    def __init__(self, data_file, urq_dir, tensor_dir, split, num_samples, batch_size):
        self.batch_size = batch_size
        self.tensor_dir = tensor_dir
        self.split = split

        # Load data_file
        processor = Processor(None, data_file, None)
        df = processor.load_pair_list()

        if num_samples is not None:
            df = df.head(num_samples)

        print(
            "Num positive pairs:",
            len(df[df["label"] == 1]),
            "\nNum negative pairs:",
            len(df[df["label"] == 0]),
        )

        # Load urqs
        self.urqs = self.load_contact_urqs(df, urq_dir)

        # Store them into batches (list of dicts)
        num_batches = len(df) // batch_size
        self.batches = []
        for i in range(num_batches):
            batch = {'concat_id': [], 'label': [], 'urq': None}
            prot1 = df['protein1'][i*batch_size:(i+1)*batch_size].tolist()
            prot2 = df['protein2'][i*batch_size:(i+1)*batch_size].tolist()
            batch['concat_id'] = self.get_concat_ids(prot1, prot2)
            batch['label'] = torch.tensor(df['label'][i*batch_size:(i+1)*batch_size].values)
            batch['urq'] = self.get_contact_urqs_batch(batch['concat_id'])
            self.batches.append(batch)

        os.makedirs(f"{tensor_dir}/batched_{split}/", exist_ok=True)

        # Make batch pt files of the logits tensors
        for i, batch in enumerate(self.batches):
            concat_ids = batch['concat_id']
            tensors = [torch.load(f"{tensor_dir}/{c}_fastLogits.pt") for c in concat_ids]
            batch_tensors = torch.cat(tensors, dim=0)
            torch.save(batch_tensors, f"{tensor_dir}/batched_{split}/batch_{i}.pt")

    def get_concat_ids(self, prot1, prot2):
        concat_ids = list(map(lambda a, b: f"{a}-{b}".translate(
            str.maketrans({'_': '-', '-': '_'})
        ), prot1, prot2))
        return concat_ids

    def load_contact_urqs(self, data, urq_dir):
        urqs = {}

        for index, row in data.iterrows():
            concat_id = row['protein1'] + '-' + row['protein2']
            concat_id = concat_id.translate(str.maketrans({'_': '-', '-': '_'}))

            if(row['label']):
                urq_path = f"{urq_dir}/{concat_id}.pt"
                urqs[concat_id] = torch.load(urq_path) if(os.path.exists(urq_path)) else torch.empty(512, 512).fill_(float('nan'))
            else:
                urqs[concat_id] = torch.zeros(512, 512)

        return urqs

    def get_contact_urqs_batch(self, ids):
        urq_batch = [self.urqs[key] for key in ids]
        urq_batch = torch.stack(urq_batch)
        return urq_batch

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        # Take the element of the list of dicts and append it with logit tensors from the batch pt file
        item = self.batches[idx]
        item['logits_tensors'] = torch.load(f"{self.tensor_dir}/batched_{self.split}/batch_{idx}.pt")
        return item

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
        concat_type: str = "gLM2"
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
    
        self.tensor_dir = tensor_dir

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = URQBatchDataset(
                self.train_file,
                self.urq_folder,
                self.tensor_dir,
                "train",
                self.num_samples,
                self.batch_size
            )
            self.val_dataset = URQBatchDataset(
                self.val_file,
                self.urq_folder,
                self.tensor_dir,
                "validate",
                self.num_samples,
                self.batch_size
            )

        if stage == "test":
            self.test_dataset = URQBatchDataset(
                self.test_file,
                self.urq_folder,
                self.tensor_dir,
                "test",
                self.num_samples,
                self.batch_size
            )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=1,
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
