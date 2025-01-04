import torch.nn as nn
import torch

from models.reveal.model import MetricLearningModel

class InterpretModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MetricLearningModel(input_dim=200, hidden_dim=256)
        self.model.load_state_dict(torch.load('../vul4c_dataset_best_f1.model'))
        self.model.eval()
        self.model.to('cuda')


    def forward(self,x):
        x = x.sum(1)
        probs, _, _ = self.model(x)
        return probs
