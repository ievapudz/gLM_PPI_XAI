from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
from data_process.Processor import Processor
from pathlib import Path

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

        print(self.data["label"])

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
        # TODO: may be unnecessary - could be removed in that case
        row['concat_id'] = pair_id
        row['sequence'] = seq
        row['length1'] = len1
        row['length2'] = len2
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
    ):
        super().__init__()
        self.fasta_file = Path(fasta_file)
        self.data_folder = Path(data_folder)
        self.test_file = self.data_folder / "test.txt"
        self.batch_size = batch_size
        self.positive_only = positive_only
        self.num_workers = num_workers
        self.num_samples = num_samples

    def setup(self, stage=None):
        if stage == "test":
            self.test_dataset = SequencePairDataset(
                self.fasta_file,
                self.test_file,
                self.num_samples,
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )