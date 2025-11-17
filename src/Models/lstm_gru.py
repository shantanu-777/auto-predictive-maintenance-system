# src/models/lstm_gru.py
import torch
import torch.nn as nn

class SequenceRegressorBase(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout

    def _init_hidden(self, batch_size, device):
        h_size = (self.num_layers * self.num_directions, batch_size, self.hidden_dim)
        c_size = (self.num_layers * self.num_directions, batch_size, self.hidden_dim)
        return (torch.zeros(h_size, device=device), torch.zeros(c_size, device=device))

class LSTMRegressor(SequenceRegressorBase):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__(input_dim, hidden_dim, num_layers, dropout, bidirectional)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, (hn, cn) = self.lstm(x)  # out: (B, T, D*H)
        # get last timestep
        last = out[:, -1, :]
        return self.head(last).squeeze(1)

class GRURegressor(SequenceRegressorBase):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__(input_dim, hidden_dim, num_layers, dropout, bidirectional)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        out, hn = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(1)
