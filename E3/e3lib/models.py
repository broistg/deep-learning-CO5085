"""
LSTM and GRU for sequence modeling. This module provides implementations of LSTM and GRU layers that can be used to solving image classification in E3.
"""

from torch import nn

from .tokenizers import make_tokenizer


class LSTMClassifier(nn.Module):
    """
    LSTM for image classification. This class implements an LSTM layer that can be used to process sequences of tokens obtained from images.
    """

    def __init__(self, 
                 tokenizer="patch4", 
                 dim=64, 
                 hidden_dim=64, 
                 num_layers=1, 
                 dropout=0.0, 
                 num_classes=10, 
                 img_size=32
        ):
        super().__init__()
        self.tokenizer = make_tokenizer(tokenizer, dim, img_size)
        self.lstm = nn.LSTM(
            input_size=dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=False
        )
        self.head = nn.Linear(
            in_features=hidden_dim, 
            out_features=num_classes
        )

    def forward(self, x):
        # [B, 3, 32, 32] -> [B, T, embed_dim]
        tokens = self.tokenizer(x)

        # Process the token sequence through the LSTM
        # [B, T, embed_dim] -> [B, T, hidden_dim]
        _, (hn, _) = self.lstm(tokens)
        features = hn[-1]  # Use the last hidden state for classification

        return self.head(features)  # Use the last hidden state for classification


class GRUClassifier(nn.Module):
    """
    GRU for image classification. This class implements a GRU layer that can be used to process sequences of tokens obtained from images.
    """
    def __init__(
        self, 
        tokenizer="patch4", 
        dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
        num_classes=10,
        img_size=32
    ):
        super().__init__()
        self.tokenizer = make_tokenizer(tokenizer, dim, img_size)
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.head = nn.Linear(
            in_features=hidden_dim,
            out_features=num_classes
        )

    def forward(self, x):
        # [B, 3, 32, 32] -> [B, T, embed_dim]
        tokens = self.tokenizer(x)

        # Process the token sequence through the GRU
        # [B, T, embed_dim] -> [B, T, hidden_dim]
        _, hn = self.gru(tokens)
        features = hn[-1]  # Use the last hidden state for classification

        return self.head(features)  # Use the last hidden state for classification


