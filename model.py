import torch
import torch.nn as nn
import warnings

try:
    from mamba_ssm import Mamba  # type: ignore
except Exception as e:  # pragma: no cover
    warnings.warn(
        'mamba-ssm is not available. Install with:\n'
        '  pip install mamba-ssm\n'
        '  pip install "mamba-ssm[causal-conv1d]"  # for CUDA conv kernels\n'
        'and ensure a compatible PyTorch/CUDA environment.',
        RuntimeWarning,
    )
    Mamba = None  # fallback marker


class MambaForecaster(nn.Module):
    """
    Multi-step forecaster using Mamba blocks.
    Input:  x [B, T, D_in]
    Output: yhat [B, H, D_out]
    """

    def __init__(
        self,
        d_in: int = 1,
        d_model: int = 128,
        n_layers: int = 2,
        horizon: int = 24,
        d_out: int | None = None,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                'mamba-ssm is not installed. Please install with:\n'
                '  pip install mamba-ssm\n  pip install "mamba-ssm[causal-conv1d]"\n'
                'and ensure a compatible PyTorch/CUDA environment.'
            )
        self.horizon = horizon
        d_out = d_out or d_in
        self.d_out = d_out

        self.in_proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon * d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in]
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        last = h[:, -1, :]
        yhat = self.head(last)
        yhat = yhat.view(x.size(0), self.horizon, self.d_out)
        return yhat


__all__ = ['MambaForecaster']
