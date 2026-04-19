import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextBiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)

        # h_n shape: [num_layers * num_directions, batch, hidden_dim]
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        out = torch.cat([forward_last, backward_last], dim=1)
        return self.dropout(out)


class MultimodalVerifier(nn.Module):
    """
    Image features come from BLIP vision encoder.
    Text sequence is encoded by BiLSTM over: question + [SEP] + candidate answer.
    """
    def __init__(
        self,
        vocab_size: int,
        image_dim: int = 768,
        proj_dim: int = 256,
        text_emb_dim: int = 128,
        text_hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.text_encoder = TextBiLSTMEncoder(
            vocab_size=vocab_size,
            emb_dim=text_emb_dim,
            hidden_dim=text_hidden_dim,
            dropout=dropout,
        )

        text_dim = self.text_encoder.output_dim
        if text_dim != proj_dim:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.text_proj = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, image_feats: torch.Tensor, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        img = self.image_proj(image_feats)
        txt = self.text_proj(self.text_encoder(input_ids, lengths))

        joint = torch.cat([img, txt, img * txt], dim=1)
        logits = self.classifier(joint)
        return logits