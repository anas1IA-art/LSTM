
from .BaseModel import BaseModel
from torch import nn
import torch

class RNNModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(RNNModel, self).__init__(vocab_size, embedding_dim, hidden_dim, n_layers, dropout)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)