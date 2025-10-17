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
        # physics options
        enable_physics: bool = False,
        n_neighbors: int = 1,
        lambda_phys: float = 0.0,
        dt: float = 20.0,
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
        # physics params
        self.enable_physics = enable_physics
        self.lambda_phys = float(lambda_phys)
        self.dt = float(dt)
        if enable_physics:
            # Learnable positive parameters via softplus on raw
            self.C_raw = nn.Parameter(torch.tensor(1.0))
            self.U_nei_raw = nn.Parameter(torch.ones(n_neighbors) * 0.1)
            self.U_corr_raw = nn.Parameter(torch.tensor(0.1))
            self.softplus = nn.Softplus()

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

    def physics_loss(self, yhat: torch.Tensor, neighbors: torch.Tensor | None, env: torch.Tensor | None, last_hist_target: torch.Tensor | None = None) -> torch.Tensor:
        """
        Discrete-time physics residual:
          C * dT/dt = sum_j U_rj * (T_nei_j - T_hat) + U_rc * (T_env - T_hat)
        where dT/dt ≈ (T_hat[k] - T_prev[k]) / dt.
        If last_hist_target is provided, it is used as T_prev for k=0; otherwise uses T_hat[k-1].

        yhat: [B, H]  (assumes d_out=1 and squeezed)
        neighbors: [B, H, Nn] or None
        env: [B, H] or None
        """
        if not self.enable_physics or self.lambda_phys <= 0.0:
            return torch.tensor(0.0, device=yhat.device)
        sp = self.softplus
        C = sp(self.C_raw)
        U_nei = sp(self.U_nei_raw)  # [Nn]
        U_corr = sp(self.U_corr_raw)

        B, H = yhat.shape
        # compute dT/dt
        if H >= 2:
            prev = torch.cat([yhat[:, :1], yhat[:, :-1]], dim=1)  # use previous predicted step for k>=1
            if last_hist_target is not None:
                prev[:, 0] = last_hist_target  # use history last target for k=0
        else:  # H == 1
            if last_hist_target is None:
                return torch.tensor(0.0, device=yhat.device)
            prev = last_hist_target.unsqueeze(1)
        dT = (yhat - prev) / max(self.dt, 1e-6)

        # flux from neighbors
        flux_nei = 0.0
        if neighbors is not None and neighbors.numel() > 0:
            # neighbors: [B,H,Nn]
            # sum over neighbors with weights U_nei
            flux_nei = (U_nei.view(1, 1, -1) * (neighbors - yhat.unsqueeze(-1))).sum(dim=-1)  # [B,H]

        flux_env = 0.0
        if env is not None and env.numel() > 0:
            flux_env = U_corr * (env - yhat)  # [B,H]

        F = C * dT - (flux_nei + flux_env)  # [B,H]
        return (F.pow(2).mean())

    def train_one_epoch(self, train_loader, optimizer, crit_data, target_channel_index: int | None = None) -> float:
        self.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            if len(batch) == 3:
                xb, yb, nb = batch
            else:
                xb, yb = batch
                nb = None
            xb = xb.to(next(self.parameters()).device)
            yb = yb.to(next(self.parameters()).device)
            optimizer.zero_grad()
            yhat = self.forward(xb).squeeze(-1)
            data_loss = crit_data(yhat, yb)
            phys_loss = torch.tensor(0.0, device=yb.device)
            if self.enable_physics and nb is not None:
                # Deduce neighbors/env by parameter shapes
                nnb = int(self.U_nei_raw.numel())
                total_cols = nb.shape[-1]
                neighbors = None
                env = None
                if total_cols == nnb:
                    neighbors = nb
                elif total_cols == nnb + 1:
                    neighbors = nb[:, :, :nnb]
                    env = nb[:, :, nnb]
                elif total_cols > 0 and nnb == 0:
                    # treat all as env if no neighbors configured
                    env = nb.squeeze(-1) if nb.shape[-1] == 1 else None
                # last history target from xb at last time step and target channel index
                last_hist_target = None
                if target_channel_index is not None:
                    last_hist_target = xb[:, -1, target_channel_index]
                phys_loss = self.physics_loss(yhat, neighbors, env, last_hist_target)
            loss = data_loss + self.lambda_phys * phys_loss
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        return total / max(n, 1)


__all__ = ['MambaForecaster']
