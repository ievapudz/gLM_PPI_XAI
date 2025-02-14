from pytorch_lightning import LightningModule
from torch import nn

class CategoricalJacobian(nn.Module):
    def __init__(self, fast: bool):
        super().__init__()
        self.fast = fast 

    def forward(self, x):
        print(x['sequence'], x['label'])
        return


class gLM2(LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.predict_step_outputs = []
        self.save_hyperparameters()

    def get_log_outputs(self, x):
        output = self.model(x)

    def step(self, batch, batch_idx, split):
        self.model.forward(batch)
        return

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

