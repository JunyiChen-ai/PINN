import argparse
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import warnings
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from preprocess import load_timeseries, build_window_samples
from model import MambaForecaster
from baseline import BaselineForecaster, resolve_baseline_from_name
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
                logging.warning(f'Non-finite in batch inputs: X={x_bad} Y={y_bad} at step {step}')
                warned = True
        yhat = model(xb).squeeze(-1)
        if not torch.isfinite(yhat).all():
            if not warned:
                yh_bad = (~torch.isfinite(yhat)).sum().item()
                logging.warning(f'Non-finite in model outputs: Yhat={yh_bad} at step {step}')
                warned = True
        loss = crit(yhat, yb)
        if not torch.isfinite(loss):
            logging.warning(f'Non-finite loss at step {step}: {loss.item()}')
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
                logging.warning(f'Non-finite in eval batch inputs: X={x_bad} Y={y_bad} at step {step}')
                warned = True
        yhat = model(xb).squeeze(-1)
        if not torch.isfinite(yhat).all():
            if not warned:
                yh_bad = (~torch.isfinite(yhat)).sum().item()
                logging.warning(f'Non-finite in eval model outputs: Yhat={yh_bad} at step {step}')
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
    # Setup logging to file Flashover/log/{model}_{dataset}_{timestamp}.log and to console
    log_dir = Path('log')
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = log_dir / f"{args.model}_{args.dataset}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")
    # Optionally enable FP32 high precision matmul; warn if not effective
    if args.enable_fp32_high_precision:
        try:
            get_prec = getattr(torch, 'get_float32_matmul_precision', None)
            before = get_prec() if callable(get_prec) else None
            torch.set_float32_matmul_precision("high")
            after = get_prec() if callable(get_prec) else None
            if after != 'high':
                logging.warning(
                    f'Requested --enable_fp32_high_precision but precision is "{after}" (was "{before}").'
                )
        except Exception as e:
            logging.warning(f'Failed to enable FP32 high precision matmul: {e}')

    if args.dataset == 3:
        data_path = Path(args.data_path) if args.data_path else default_excel_path(Path.cwd())
        if not data_path or not Path(data_path).exists():
            logging.error('Excel data file not found. Set --data_path or env FLASHOVER_XLSX.')
            raise FileNotFoundError('Excel data file not found. Set --data_path or env FLASHOVER_XLSX.')
        channels = tuple([c.strip() for c in args.channels.split(',') if c.strip()])
        neighbor_chs = tuple([c.strip() for c in args.neighbor_channels.split(',') if c.strip()])
        env_ch = args.env_channel.strip() if args.env_channel else None
        future_chs = tuple([*neighbor_chs, *( [env_ch] if env_ch else [] )])
        # caching under three-compartments folder
        cache_dir = data_path.parent
        cache_name = (
            f"cache_h{args.history}_H{args.horizon}_s{args.stride}_t{args.trunc_temp if args.trunc_temp is not None else 'None'}_"
            f"chs{','.join(channels)}_tgt{args.target_channel}_nei{','.join(neighbor_chs)}_env{env_ch if env_ch else 'None'}_"
            f"tb{int(bool(args.train_balance))}_te{int(bool(args.test_balance))}.pt"
        ).replace(' ', '')
        cache_path = cache_dir / cache_name
        if cache_path.exists() and not args.rebuild_cache:
            logging.info(f'Loading cached samples: {cache_path}')
            import torch as _torch
            cached = _torch.load(cache_path)
            X, Y, E, N = cached['X'], cached['Y'], cached['E'], cached['N']
            meta = cached.get('meta', {})
            if meta:
                logging.info(f"Loaded cache meta: {meta}")
        else:
            series = load_timeseries(str(data_path))
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
            try:
                import torch as _torch
                _torch.save({'X': X, 'Y': Y, 'E': E, 'N': N,
                             'meta': {
                                 'dataset': 3,
                                 'history': args.history,
                                 'horizon': args.horizon,
                                 'stride': args.stride,
                                 'trunc_temp': args.trunc_temp,
                                 'channels': list(channels),
                                 'target_channel': args.target_channel,
                                 'neighbors': list(neighbor_chs),
                                 'env': env_ch,
                                 'train_balance': bool(args.train_balance),
                                 'test_balance': bool(args.test_balance),
                             }}, cache_path)
                logging.info(f'Saved samples cache to {cache_path}')
            except Exception as e:
                logging.warning(f'Failed to save cache: {e}')
        auto_target_idx = channels.index(args.target_channel) if args.target_channel in channels else None
        n_neighbors = len(neighbor_chs)
        d_in = len(channels)
    else:
        # six-compartment dataset: auto target/neighbors per experiment, env left empty, with caching
        from preprocess import load_six_series, build_window_samples_dynamic
        six_root = Path('DataForFlashover/six-compartments')
        six_root_abs = Path.cwd() / six_root
        cache_dir = six_root_abs
        cache_name = (
            f"cache6_h{args.history}_H{args.horizon}_s{args.stride}_t{args.trunc_temp if args.trunc_temp is not None else 'None'}_"
            f"tb{int(bool(args.train_balance))}_te{int(bool(args.test_balance))}.pt"
        )
        cache_path = cache_dir / cache_name
        if cache_path.exists() and not args.rebuild_cache:
            logging.info(f'Loading cached samples (six): {cache_path}')
            import torch as _torch
            cached = _torch.load(cache_path)
            X, Y, E, N = cached['X'], cached['Y'], cached['E'], cached['N']
            meta = cached.get('meta', {})
            if meta:
                logging.info(f"Loaded cache meta (six): {meta}")
        else:
            series, exp_to_room, room_to_hd = load_six_series(str(six_root_abs))
            # Build mapping exp->target HD, neighbors (default: all others)
            exp_to_target: dict[int, str] = {}
            exp_to_neighbors: dict[int, list[str]] = {}
            for exp, room_num in exp_to_room.items():
                code_to_name = {'1':'Dining Room','2':'Kitchen','4':'Living Room','5':'Bedroom 1','7':'Bedroom 2','8':'Bedroom 3'}
                name = code_to_name.get(room_num, None)
                if name and name in room_to_hd:
                    tgt = room_to_hd[name]
                else:
                    tgt = 'HD1'
                exp_to_target[exp] = tgt
                all_hds = ['HD1','HD2','HD3','HD4','HD5','HD6']
                exp_to_neighbors[exp] = [h for h in all_hds if h != tgt]
            X, Y, E, N = build_window_samples_dynamic(
                series=series,
                exp_to_target_hd=exp_to_target,
                exp_to_neighbors=exp_to_neighbors,
                history=args.history,
                horizon=args.horizon,
                stride=args.stride,
                trunc_temp=args.trunc_temp,
                stop_when_all_channels_reach_trunc=True if args.trunc_temp is not None else False,
            )
            try:
                import torch as _torch
                _torch.save({'X': X, 'Y': Y, 'E': E, 'N': N,
                             'meta': {
                                 'dataset': 6,
                                 'history': args.history,
                                 'horizon': args.horizon,
                                 'stride': args.stride,
                                 'trunc_temp': args.trunc_temp,
                                 'train_balance': bool(args.train_balance),
                                 'test_balance': bool(args.test_balance),
                             }}, cache_path)
                logging.info(f'Saved samples cache (six) to {cache_path}')
            except Exception as e:
                logging.warning(f'Failed to save cache (six): {e}')
        channels = tuple()  # not used for six in dynamic mode
        auto_target_idx = None
        n_neighbors = 5  # all other rooms as neighbors
        d_in = 6
    if not X:
        logging.error('No samples built. Try reducing --history/--horizon or disable truncation.')
        raise RuntimeError('No samples built. Try reducing --history/--horizon or disable truncation.')

    # Print dataset-level class distribution for current config
    total = len(Y)
    pos = sum(1 for y in Y if (torch.tensor(y) >= args.threshold).any().item())
    neg = total - pos
    trunc_note = f" (trunc_temp={args.trunc_temp})" if args.trunc_temp is not None else ""
    logging.info(f'Dataset windows: total={total}  pos={pos}  neg={neg}  ratio={pos/total if total else 0:.4f}{trunc_note}')

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
        logging.info(f'[Fold {k+1}/5] test windows: total={ttotal}  pos={tpos}  neg={tneg}  ratio={tpos/ttotal if ttotal else 0:.4f}{trunc_note}')
        # Optional balancing of test set by downsampling to the minority count
        balanced_test_idx = test_idx
        if args.test_balance and ttotal > 0 and tpos > 0 and tneg > 0:
            # Partition test indices by label
            pos_ids = []
            neg_ids = []
            for idx in test_idx:
                y = Y[idx]
                is_pos = (torch.tensor(y) >= args.threshold).any().item()
                (pos_ids if is_pos else neg_ids).append(idx)
            m = min(len(pos_ids), len(neg_ids))
            if m > 0:
                rng = random.Random(args.seed + k)
                rng.shuffle(pos_ids)
                rng.shuffle(neg_ids)
                balanced_test_idx = pos_ids[:m] + neg_ids[:m]
                # Shuffle combined to avoid order bias
                rng.shuffle(balanced_test_idx)
                logging.info(f'[Fold {k+1}/5] balanced test windows: total={2*m}  pos={m}  neg={m}')
        # Optional train balancing similarly
        balanced_train_idx = train_idx
        if args.train_balance:
            # compute labels for train samples
            tpos_ids = []
            tneg_ids = []
            for idx in train_idx:
                y = Y[idx]
                is_pos = (torch.tensor(y) >= args.threshold).any().item()
                (tpos_ids if is_pos else tneg_ids).append(idx)
            m = min(len(tpos_ids), len(tneg_ids))
            if m > 0:
                rng2 = random.Random(args.seed + 100 + k)
                rng2.shuffle(tpos_ids)
                rng2.shuffle(tneg_ids)
                balanced_train_idx = tpos_ids[:m] + tneg_ids[:m]
                rng2.shuffle(balanced_train_idx)
                logging.info(f'[Fold {k+1}/5] balanced train windows: total={2*m}  pos={m}  neg={m}')
        # Apply the same clamping behavior to both train and test inputs when truncation is enabled
        train_ds = WindowDataset(X, Y, N, indices=balanced_train_idx, clip_max=args.trunc_temp)
        test_ds = WindowDataset(X, Y, N, indices=balanced_test_idx, clip_max=args.trunc_temp)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

        if args.model.lower() == 'mamba':
            model = MambaForecaster(
                d_in=d_in,
                d_model=args.d_model,
                n_layers=args.n_layers,
                horizon=args.horizon,
                d_state=args.d_state,
                d_conv=args.d_conv,
                expand=args.expand,
                d_out=1,
                enable_physics=args.enable_physics_loss,
                n_neighbors=n_neighbors,
                lambda_phys=args.lambda_phys,
                dt=args.dt,
            ).to(device)
        else:
            rnn_type, bidir, use_attn = resolve_baseline_from_name(args.model)
            model = BaselineForecaster(
                d_in=d_in,
                horizon=args.horizon,
                rnn_type=rnn_type,
                hidden_size=args.hidden_size,
                num_layers=args.rnn_layers,
                bidirectional=bidir,
                use_attention=use_attn,
                dropout=args.dropout,
            ).to(device)

        crit = nn.L1Loss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_mae = float('inf')
        best_sup0 = 0
        best_sup1 = 0
        best_ep = 0
        best_rep = ''
        # checkpoint dir
        ckpt_dir = Path('checkpoint') / str(args.dataset) / f'fold-{k+1}' / args.model
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / 'best.pt'

        for ep in range(1, args.epochs + 1):
            train_loss = model.train_one_epoch(train_loader, opt, crit, auto_target_idx)
            # Evaluate each epoch on the test set
            m_mae, sup0, sup1, cls_rep = evaluate(model, test_loader, device, threshold=args.threshold)
            improved = m_mae < best_mae
            if improved:
                best_mae, best_sup0, best_sup1, best_ep, best_rep = m_mae, sup0, sup1, ep, cls_rep
                # save best checkpoint for this fold
                try:
                    torch.save({'epoch': ep,
                                'state_dict': model.state_dict(),
                                'val_mae': m_mae,
                                'args': vars(args)}, ckpt_path)
                    logging.info(f'[Fold {k+1}/5] Saved best checkpoint to {ckpt_path}')
                except Exception as e:
                    logging.warning(f'Failed to save checkpoint: {e}')
            # Compact per-epoch log
            logging.info(f'[Fold {k+1}/5][Epoch {ep}/{args.epochs}] train L1: {train_loss:.4f}  val MAE: {m_mae:.4f}  Support: neg={sup0} pos={sup1}' + ('  [improved]' if improved else ''))
            if improved:
                logging.info(f'[Fold {k+1}/5][Epoch {ep}] Classification report (best so far):\n{cls_rep}')

        # Use best epoch within the fold as the fold result
        fold_mae.append(best_mae)
        total_support0 += best_sup0
        total_support1 += best_sup1
        logging.info(f'[Fold {k+1}/5] Best@Epoch {best_ep}: MAE={best_mae:.4f}  Support: neg={best_sup0}  pos={best_sup1}')
        logging.info(f'[Fold {k+1}/5] Best Classification report:\n{best_rep}')

    def avg(xs: List[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    logging.info('========== 5-fold CV Summary =========')
    logging.info(f'MAE:  mean={avg(fold_mae):.4f}  folds={[f"{v:.4f}" for v in fold_mae]}')
    logging.info(f'Support (total across folds): neg={total_support0}  pos={total_support1}')


if __name__ == '__main__':
    main()
