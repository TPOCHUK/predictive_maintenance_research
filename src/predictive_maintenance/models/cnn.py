import torch.nn as nn


class RULCNN(nn.Module):
    def __init__(self, in_features, window, dropout_p=0.3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
        )

        flat = 64 * (window - 4)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x.transpose(1, 2))
        return self.fc_layers(x).squeeze(-1)
