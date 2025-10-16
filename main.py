import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import warnings
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from preprocess import load_timeseries, build_window_samples
from model import MambaForecaster
from util import (
    set_seed,
    get_device,
    WindowDataset,
    make_folds,
    split_by_exp,
    binary_metrics,
    mae as mae_metric,
    default_excel_path,
    add_main_args,
)


def train_one_epoch(model, loader, crit, opt, device) -> float:
    model.train()
    total = 0.0
    n = 0
    warned = False
    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        if not torch.isfinite(xb).all() or not torch.isfinite(yb).all():
            if not warned:
                x_bad = (~torch.isfinite(xb)).sum().item()
                y_bad = (~torch.isfinite(yb)).sum().item()
                print(f'[WARN][train] Non-finite in batch inputs: X={x_bad} Y={y_bad} at step {step}')
                warned = True
        yhat = model(xb).squeeze(-1)
        if not torch.isfinite(yhat).all():
            if not warned:
                yh_bad = (~torch.isfinite(yhat)).sum().item()
                print(f'[WARN][train] Non-finite in model outputs: Yhat={yh_bad} at step {step}')
                warned = True
        loss = crit(yhat, yb)
        if not torch.isfinite(loss):
            print(f'[WARN][train] Non-finite loss at step {step}: {loss.item()}')
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float) -> Tuple[float, float, float]:
    model.eval()
    maes: List[float] = []
    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []
    warned = False
    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)
        if not torch.isfinite(xb).all() or not torch.isfinite(yb).all():
            if not warned:
                x_bad = (~torch.isfinite(xb)).sum().item()
                y_bad = (~torch.isfinite(yb)).sum().item()
                print(f'[WARN][eval] Non-finite in batch inputs: X={x_bad} Y={y_bad} at step {step}')
                warned = True
        yhat = model(xb).squeeze(-1)
        if not torch.isfinite(yhat).all():
            if not warned:
                yh_bad = (~torch.isfinite(yhat)).sum().item()
                print(f'[WARN][eval] Non-finite in model outputs: Yhat={yh_bad} at step {step}')
                warned = True
        maes.append(mae_metric(yhat, yb))
        # binary labels: any temp >= threshold in horizon (per sample)
        yt = (yb >= threshold).any(dim=1).to(torch.int)
        yp = (yhat >= threshold).any(dim=1).to(torch.int)
        y_true_bin.extend(yt.cpu().tolist())
        y_pred_bin.extend(yp.cpu().tolist())
    acc, macro_f1 = binary_metrics(y_true_bin, y_pred_bin)
    mean_mae = sum(maes) / max(len(maes), 1)
    return mean_mae, acc, macro_f1


def main():
    parser = argparse.ArgumentParser(description='P-Flash (no-phase) Mamba forecaster')
    add_main_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    # Optionally enable FP32 high precision matmul; warn if not effective
    if args.enable_fp32_high_precision:
        try:
            get_prec = getattr(torch, 'get_float32_matmul_precision', None)
            before = get_prec() if callable(get_prec) else None
            torch.set_float32_matmul_precision("high")
            after = get_prec() if callable(get_prec) else None
            if after != 'high':
                warnings.warn(
                    f'Requested --enable_fp32_high_precision but precision is "{after}" (was "{before}").',
                    RuntimeWarning,
                )
        except Exception as e:
            warnings.warn(
                f'Failed to enable FP32 high precision matmul: {e}',
                RuntimeWarning,
            )

    data_path = Path(args.data_path) if args.data_path else default_excel_path(Path.cwd())
    if not data_path or not Path(data_path).exists():
        raise FileNotFoundError('Excel data file not found. Set --data_path or env FLASHOVER_XLSX.')

    series = load_timeseries(str(data_path))
    channels = tuple([c.strip() for c in args.channels.split(',') if c.strip()])
    X, Y, E = build_window_samples(
        series=series,
        history=args.history,
        horizon=args.horizon,
        stride=args.stride,
        use_channels=channels,
        target_channel=args.target_channel,
        trunc_temp=None,
        drop_after_fail=False,
    )
    if not X:
        raise RuntimeError('No samples built. Try reducing --history/--horizon or disable truncation.')

    # Print dataset-level class distribution for current config
    total = len(Y)
    pos = sum(1 for y in Y if (torch.tensor(y) >= args.threshold).any().item())
    neg = total - pos
    trunc_note = f" (trunc_temp={args.trunc_temp})" if args.trunc_temp is not None else ""
    print(f'[Info] Dataset windows: total={total}  pos={pos}  neg={neg}  ratio={pos/total if total else 0:.4f}{trunc_note}')

    ds = WindowDataset(X, Y)
    folds = make_folds(E, n_splits=5, seed=args.seed)

    fold_mae: List[float] = []
    fold_acc: List[float] = []
    fold_f1: List[float] = []

    for k in range(5):
        # Fold-level distribution (test split)
        train_idx, test_idx = split_by_exp(E, folds, k)
        tY = [Y[i] for i in test_idx]
        ttotal = len(tY)
        tpos = sum(1 for y in tY if (torch.tensor(y) >= args.threshold).any().item())
        tneg = ttotal - tpos
        print(f'[Info][Fold {k+1}/5] test windows: total={ttotal}  pos={tpos}  neg={tneg}  ratio={tpos/ttotal if ttotal else 0:.4f}{trunc_note}')
        train_ds = WindowDataset(X, Y, indices=train_idx, clip_max=args.trunc_temp)
        test_ds = WindowDataset(X, Y, indices=test_idx, clip_max=None)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = MambaForecaster(
            d_in=len(channels),
            d_model=args.d_model,
            n_layers=args.n_layers,
            horizon=args.horizon,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            d_out=1,  # predicting target_channel scalar trajectory
        ).to(device)

        crit = nn.L1Loss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        for ep in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, crit, opt, device)
            # simple progress print per epoch
            print(f'[Fold {k+1}/5][Epoch {ep}/{args.epochs}] train L1: {train_loss:.4f}')

        m_mae, m_acc, m_f1 = evaluate(model, test_loader, device, threshold=args.threshold)
        fold_mae.append(m_mae)
        fold_acc.append(m_acc)
        fold_f1.append(m_f1)
        print(f'[Fold {k+1}/5] MAE={m_mae:.4f}  Acc={m_acc:.4f}  MacroF1={m_f1:.4f}')

    def avg(xs: List[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    print('========== 5-fold CV Summary =========')
    print(f'MAE:  mean={avg(fold_mae):.4f}  folds={[f"{v:.4f}" for v in fold_mae]}')
    print(f'Acc:  mean={avg(fold_acc):.4f}  folds={[f"{v:.4f}" for v in fold_acc]}')
    print(f'F1 :  mean={avg(fold_f1):.4f}  folds={[f"{v:.4f}" for v in fold_f1]}')


if __name__ == '__main__':
    main()
