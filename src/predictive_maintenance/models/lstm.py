import torch.nn as nn


class RULLSTM(nn.Module):
    """Predict RUL from the last LSTM state."""

    def __init__(self, in_features, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)
