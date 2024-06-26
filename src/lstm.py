import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_epochs, batch_size, learning_rate):
        super().__init__()
        self.Embedding = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=0.5)
        self.Linear = nn.Linear(hidden_dim, output_dim)
        self.Sigmoid = nn.Sigmoid()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def forward(self, x, lengths):
        embedded = self.Embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        pack_padded_output, _ = self.LSTM(packed_embedded)
        pad_packed_output, _ = pad_packed_sequence(pack_padded_output, batch_first=True)
        last_hidden_state = pad_packed_output[:, -1, :]
        final_state = self.Linear(last_hidden_state)
        output = self.Sigmoid(final_state)
        return output
