from torch import nn
import torch
import pathlib
from transformers import AutoTokenizer, AutoModel
import os

TOKENIZERS_PARALLELISM = True

class EmbeddingsGenerator(nn.Module):
    def __init__(self, emb_path: str):
        super().__init__()
        self.emb_dim = 1280
        self.pool = "mean"
        self.embeddings = None
        self.emb_path = emb_path
        pathlib.Path(self.emb_path).mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_path = "./gLM2_650M"
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.MASK_TOKEN_ID = self.tokenizer.mask_token_id

        for param in self.model.parameters(): param.requires_grad = False

    def forward(self, batch):
        embeddings = torch.empty((len(batch['concat_id']), self.emb_dim)).to(self.device)

        for i, concat_id in enumerate(batch['concat_id']):
            embedding_path = os.path.join(self.emb_path, f'{concat_id}.pt')
            if not os.path.isfile(embedding_path):
                encoding = self.tokenizer(batch['sequence'][i], return_tensors='pt')
                with torch.no_grad():
                    embedding = self.model(encoding.input_ids.to(self.device), output_hidden_states=True).last_hidden_state
                    torch.save(embedding, os.path.join(self.emb_path, f'{concat_id}.pt'))
        #return embeddings
