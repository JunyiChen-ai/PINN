import torch
import torch.nn as nn
from typing import Optional


class BaselineForecaster(nn.Module):
    """
    RNN/LSTM/GRU baselines with optional bidirection and attention.

    Input:  x [B, T, D]
    Output: yhat [B, H, 1]
    """

    def __init__(
        self,
        d_in: int,
        horizon: int,
        rnn_type: str = 'lstm',  # 'rnn'|'lstm'|'gru'
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = False,
        use_attention: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.use_attention = use_attention
        self.bidirectional = bidirectional

        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}.get(rnn_type.lower())
        if rnn_cls is None:
            raise ValueError(f'Unsupported rnn_type: {rnn_type}')

        self.rnn = rnn_cls(
            input_size=d_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        rep_dim = hidden_size * (2 if bidirectional else 1)

        if use_attention:
            # Simple additive attention over time: score_t = v^T tanh(W h_t + b)
            self.attn_W = nn.Linear(rep_dim, rep_dim)
            self.attn_v = nn.Linear(rep_dim, 1, bias=False)
        else:
            self.attn_W = None
            self.attn_v = None

        self.head = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.GELU(),
            nn.Linear(rep_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        out, _ = self.rnn(x)
        # out: [B, T, H*dir]
        if self.use_attention:
            # scores: [B,T,1]
            scores = self.attn_v(torch.tanh(self.attn_W(out)))
            weights = torch.softmax(scores, dim=1)  # across time
            rep = (weights * out).sum(dim=1)  # [B, H*dir]
        else:
            # take last time step representation
            rep = out[:, -1, :]
        yhat = self.head(rep)  # [B, H]
        return yhat.unsqueeze(-1)  # [B, H, 1]

    def train_one_epoch(self, train_loader, optimizer, crit_data, target_channel_index: Optional[int] = None) -> float:
        self.train()
        total = 0.0
        n = 0
        device = next(self.parameters()).device
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, _ = batch
            else:
                xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            yhat = self.forward(xb).squeeze(-1)
            loss = crit_data(yhat, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        return total / max(n, 1)


def resolve_baseline_from_name(name: str):
    """
    Map friendly names to (rnn_type, bidirectional, use_attention).
    Supported names:
      'RNN', 'LSTM', 'GRU',
      'Bi-RNN', 'Bi-LSTM', 'Bi-GRU',
      'Bi-LSTM-Attention', 'Bi-GRU-Attention'
    """
    key = name.lower()
    if key == 'rnn':
        return ('rnn', False, False)
    if key == 'lstm':
        return ('lstm', False, False)
    if key == 'gru':
        return ('gru', False, False)
    if key in ('bi-rnn', 'birnn'):
        return ('rnn', True, False)
    if key in ('bi-lstm', 'bilstm'):
        return ('lstm', True, False)
    if key in ('bi-gru', 'bigru'):
        return ('gru', True, False)
    if key in ('bi-lstm-attention', 'bilstm-attention'):
        return ('lstm', True, True)
    if key in ('bi-gru-attention', 'bigru-attention'):
        return ('gru', True, True)
    raise ValueError(f'Unknown baseline model name: {name}')


__all__ = ['BaselineForecaster', 'resolve_baseline_from_name']

