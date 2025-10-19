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
    # Special case for gFlashNet: return sentinel
    if key in ('gflashnet', 'gflash'):
        return (None, False, False)
    if key in ('cnn-lstm', 'cnnlstm', 'conv-lstm', 'convlstm'):
        return ('cnn_lstm', False, False)
    raise ValueError(f'Unknown baseline model name: {name}')


__all__ = ['BaselineForecaster', 'resolve_baseline_from_name']

# ================== gFlashNet for temperature forecasting ==================

import math
from typing import Tuple


class TemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation)
        self.res = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, C, T]
        z = self.act(self.conv(x))
        r = self.res(x)
        return z + r


class GATHead(nn.Module):
    def __init__(self, c_in: int, c_out: int, negative_slope: float = 0.2):
        super().__init__()
        self.lin = nn.Linear(c_in, c_out, bias=False)
        self.attn = nn.Parameter(torch.empty(size=(2 * c_out, 1)))
        nn.init.xavier_uniform_(self.attn)
        self.leaky = nn.LeakyReLU(negative_slope)

    def forward(self, x_ntc: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x_ntc: [B, N, C]; A: [N, N] (0/1)
        B, N, C = x_ntc.shape
        Wh = self.lin(x_ntc)  # [B,N,C']
        # Compute attention scores e_ij = a^T [Wh_i || Wh_j]
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, -1)
        cat = torch.cat([Wh_i, Wh_j], dim=-1)  # [B,N,N,2C']
        e = self.leaky(torch.matmul(cat, self.attn).squeeze(-1))  # [B,N,N]
        # Mask using adjacency (no self-loop)
        mask = (A > 0).unsqueeze(0).expand_as(e)
        e = e.masked_fill(~mask, float('-inf'))
        alpha = torch.softmax(e, dim=-1)  # along neighbor dim j
        out = torch.matmul(alpha, Wh)  # [B,N,C']
        return out


class SpatialBlockGAT(nn.Module):
    def __init__(self, c_in: int, c_out: int, num_heads: int = 2):
        super().__init__()
        self.heads = nn.ModuleList([GATHead(c_in, c_out) for _ in range(num_heads)])
        self.proj = nn.Linear(c_out, c_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C, T] — apply GAT independently per time slice
        B, N, C, T = x.shape
        outs = []
        for t in range(T):
            xt = x[:, :, :, t].contiguous()  # [B,N,C]
            hsum = 0
            for head in self.heads:
                hsum = hsum + head(xt, A)
            hmean = hsum / len(self.heads)  # [B,N,C]
            outs.append(self.act(self.proj(hmean)))
        y = torch.stack(outs, dim=-1)  # [B,N,C,T]
        return y


class STBlock(nn.Module):
    def __init__(self, c_mid: int, kernel_size: int = 3, dil1: int = 1, dil2: int = 2, heads: int = 2):
        super().__init__()
        self.temp1 = TemporalBlock(c_mid, c_mid, kernel_size, dil1)
        self.spat = SpatialBlockGAT(c_mid, c_mid, num_heads=heads)
        self.temp2 = TemporalBlock(c_mid, c_mid, kernel_size, dil2)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B,N,C,T]
        B, N, C, T = x.shape
        z = x.reshape(B * N, C, T)
        z = self.temp1(z)
        z = z.reshape(B, N, C, T)
        z = self.spat(z, A)
        z2 = z.reshape(B * N, C, T)
        z2 = self.temp2(z2)
        z2 = z2.reshape(B, N, C, T)
        return z2


class GFlashNetForecaster(nn.Module):
    def __init__(
        self,
        d_in: int,
        horizon: int,
        c_hidden: int = 64,
        kernel_size: int = 3,
        heads: int = 2,
        dilations: Tuple[int, int, int, int] = (1, 2, 1, 2),
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.d_in = d_in
        self.c_in = 1
        self.c_hidden = c_hidden
        d1, d2, d3, d4 = dilations
        # in projection per node: implemented as 1x1 conv on time for each node independently
        self.in_proj = nn.Conv1d(self.c_in, c_hidden, kernel_size=1)
        self.st1 = STBlock(c_hidden, kernel_size, d1, d2, heads)
        self.st2 = STBlock(c_hidden, kernel_size, d3, d4, heads)
        self.head = nn.Sequential(
            nn.Linear(c_hidden, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, horizon),
        )

    def build_adjacency(self, N: int, device: torch.device) -> torch.Tensor:
        # default: unweighted, undirected full graph without self-loops
        A = torch.ones(N, N, device=device)
        A.fill_diagonal_(0.0)
        return A

    def forward(self, x: torch.Tensor, target_index: int | None = None) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        assert D == self.d_in, f'd_in mismatch: got {D}, expected {self.d_in}'
        # reshape to [B,N,C,T]
        xn = x.permute(0, 2, 1).unsqueeze(2).contiguous()  # [B,D,1,T]
        # apply per-node temporal conv via view
        Bn, N, C, Tr = xn.shape
        z = xn.reshape(Bn * N, C, Tr)
        z = self.in_proj(z)  # [B*N, C_hidden, T]
        z = z.reshape(Bn, N, self.c_hidden, Tr)
        A = self.build_adjacency(N, x.device)
        z = self.st1(z, A)
        z = self.st2(z, A)
        # select target node features (last time step)
        if target_index is None:
            target_index = N - 1  # default to last feature
        z_tgt = z[:, target_index, :, -1]  # [B, C_hidden]
        yhat = self.head(z_tgt)  # [B, H]
        return yhat.unsqueeze(-1)

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
            yhat = self.forward(xb, target_channel_index).squeeze(-1)
            loss = crit_data(yhat, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        return total / max(n, 1)


__all__.extend(['GFlashNetForecaster'])

# ================== CNN + LSTM baseline ==================


class CNNLSTMForecaster(nn.Module):
    """
    Temporal CNN encoder followed by LSTM for multi-step temperature forecasting.

    Input:  x [B, T, D]
    Output: yhat [B, H, 1]
    """

    def __init__(
        self,
        d_in: int,
        horizon: int,
        hidden_size: int = 128,
        rnn_layers: int = 2,
        kernel_size: int = 3,
        dilation: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        pad = (kernel_size - 1) * dilation // 2
        # Conv over time with feature channels as in_channels
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels=d_in, out_channels=hidden_size, kernel_size=kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        rep_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.GELU(),
            nn.Linear(rep_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> conv expects [B, C_in=D, T]
        xb = x.permute(0, 2, 1).contiguous()
        z = self.temporal(xb)  # [B, hidden, T]
        zt = z.permute(0, 2, 1).contiguous()  # [B, T, hidden]
        out, _ = self.lstm(zt)
        rep = out[:, -1, :]
        yhat = self.head(rep)  # [B, H]
        return yhat.unsqueeze(-1)

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


__all__.extend(['CNNLSTMForecaster'])
