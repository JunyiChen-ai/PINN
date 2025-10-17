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
def evaluate(model, loader, device, threshold: float) -> Tuple[float, int, int, str]:
    model.eval()
    maes: List[float] = []
    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []
    warned = False
    for step, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            xb, yb, nb = batch
        else:
            xb, yb = batch
            nb = None
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
    mean_mae = sum(maes) / max(len(maes), 1)
    # Prefer sklearn's definition of support (count of true instances per class)
    support0 = y_true_bin.count(0)
    support1 = y_true_bin.count(1)
    report_text = ''
    try:
        from sklearn.metrics import classification_report  # type: ignore
        # Produce human-readable report
        report_text = classification_report(
            y_true_bin, y_pred_bin,
            target_names=["neg", "pos"],
            digits=4,
            zero_division=0,
        )
        # Also fetch dict to get robust supports
        rep = classification_report(y_true_bin, y_pred_bin, output_dict=True, zero_division=0)
        # Fall back to counts above if keys missing
        support0 = int(rep.get('0', {}).get('support', support0))
        support1 = int(rep.get('1', {}).get('support', support1))
    except Exception:
        # Minimal fallback report
        tp = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt==1 and yp==1)
        tn = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt==0 and yp==0)
        fp = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt==0 and yp==1)
        fn = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt==1 and yp==0)
        total = tp+tn+fp+fn
        acc = (tp+tn)/total if total else 0.0
        report_text = (
            f'Fallback report (sklearn not available)\n'
            f' support neg={support0} pos={support1}\n'
            f' TP={tp} TN={tn} FP={fp} FN={fn}  Acc={acc:.4f}'
        )
    return mean_mae, support0, support1, report_text


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
    neighbor_chs = tuple([c.strip() for c in args.neighbor_channels.split(',') if c.strip()])
    env_ch = args.env_channel.strip() if args.env_channel else None
    future_chs = tuple([*neighbor_chs, *( [env_ch] if env_ch else [] )])
    X, Y, E, N = build_window_samples(
        series=series,
        history=args.history,
        horizon=args.horizon,
        stride=args.stride,
        use_channels=channels,
        target_channel=args.target_channel,
        trunc_temp=args.trunc_temp,
        drop_after_fail=False,
        stop_when_all_channels_reach_trunc=True if args.trunc_temp is not None else False,
        future_channels=future_chs,
    )
    if not X:
        raise RuntimeError('No samples built. Try reducing --history/--horizon or disable truncation.')

    # Print dataset-level class distribution for current config
    total = len(Y)
    pos = sum(1 for y in Y if (torch.tensor(y) >= args.threshold).any().item())
    neg = total - pos
    trunc_note = f" (trunc_temp={args.trunc_temp})" if args.trunc_temp is not None else ""
    print(f'[Info] Dataset windows: total={total}  pos={pos}  neg={neg}  ratio={pos/total if total else 0:.4f}{trunc_note}')

    ds = WindowDataset(X, Y, N)
    folds = make_folds(E, n_splits=5, seed=args.seed)

    fold_mae: List[float] = []
    total_support0 = 0
    total_support1 = 0

    for k in range(5):
        # Fold-level distribution (test split)
        train_idx, test_idx = split_by_exp(E, folds, k)
        tY = [Y[i] for i in test_idx]
        ttotal = len(tY)
        tpos = sum(1 for y in tY if (torch.tensor(y) >= args.threshold).any().item())
        tneg = ttotal - tpos
        print(f'[Info][Fold {k+1}/5] test windows: total={ttotal}  pos={tpos}  neg={tneg}  ratio={tpos/ttotal if ttotal else 0:.4f}{trunc_note}')
        # Apply the same clamping behavior to both train and test inputs when truncation is enabled
        train_ds = WindowDataset(X, Y, N, indices=train_idx, clip_max=args.trunc_temp)
        test_ds = WindowDataset(X, Y, N, indices=test_idx, clip_max=args.trunc_temp)
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
            enable_physics=args.enable_physics_loss,
            n_neighbors=len(neighbor_chs),
            lambda_phys=args.lambda_phys,
            dt=args.dt,
        ).to(device)

        crit = nn.L1Loss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_mae = float('inf')
        best_sup0 = 0
        best_sup1 = 0
        best_ep = 0
        best_rep = ''

        for ep in range(1, args.epochs + 1):
            target_idx = channels.index(args.target_channel) if args.target_channel in channels else None
            train_loss = model.train_one_epoch(train_loader, opt, crit, target_idx)
            # Evaluate each epoch on the test set
            m_mae, sup0, sup1, cls_rep = evaluate(model, test_loader, device, threshold=args.threshold)
            improved = m_mae < best_mae
            if improved:
                best_mae, best_sup0, best_sup1, best_ep, best_rep = m_mae, sup0, sup1, ep, cls_rep
            # Compact per-epoch log
            print(f'[Fold {k+1}/5][Epoch {ep}/{args.epochs}] train L1: {train_loss:.4f}  val MAE: {m_mae:.4f}  Support: neg={sup0} pos={sup1}' + ('  [improved]' if improved else ''))
            if improved:
                print(f'[Fold {k+1}/5][Epoch {ep}] Classification report (best so far):\n{cls_rep}')

        # Use best epoch within the fold as the fold result
        fold_mae.append(best_mae)
        total_support0 += best_sup0
        total_support1 += best_sup1
        print(f'[Fold {k+1}/5] Best@Epoch {best_ep}: MAE={best_mae:.4f}  Support: neg={best_sup0}  pos={best_sup1}')
        print(f'[Fold {k+1}/5] Best Classification report:\n{best_rep}')

    def avg(xs: List[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    print('========== 5-fold CV Summary =========')
    print(f'MAE:  mean={avg(fold_mae):.4f}  folds={[f"{v:.4f}" for v in fold_mae]}')
    print(f'Support (total across folds): neg={total_support0}  pos={total_support1}')


if __name__ == '__main__':
    main()
