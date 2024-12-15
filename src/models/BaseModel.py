from torch import nn

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(BaseModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        embeds = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embeds, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        raise NotImplementedError
