import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, text, seq_length, vocab=None):
        self.text = text
        self.seq_length = seq_length
        
        # Create vocabulary if not provided
        if vocab is None:
            # Create character-level vocabulary
            chars = sorted(list(set(text)))
            self.char2idx = {char: idx for idx, char in enumerate(chars)}
            self.idx2char = {idx: char for idx, char in enumerate(chars)}
            # Add unknown token
            self.char2idx['<UNK>'] = len(self.char2idx)
            self.idx2char[len(self.idx2char)] = '<UNK>'
        else:
            self.char2idx, self.idx2char = vocab
        
        self.vocab_size = len(self.char2idx)
        self.data = self.encode_text()
        
    def encode_text(self):
        return [self.char2idx.get(char, self.char2idx['<UNK>']) for char in self.text]
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length]
        target = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.LongTensor(sequence), torch.LongTensor(target)