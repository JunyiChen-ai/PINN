import argparse
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from baseline import BaselineForecaster, resolve_baseline_from_name
from model import MambaForecaster
from preprocess import (
    load_timeseries,
    build_window_samples,
    load_six_series,
    build_window_samples_dynamic,
)
from util import (
    WindowDataset,
    make_folds,
    split_by_exp,
    mae as mae_metric,
    set_seed,
    get_device,
)


@torch.no_grad()
def evaluate_model(model, loader, device, threshold: float) -> Tuple[float, str, int, int]:
    model.eval()
    maes: List[float] = []
    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb).squeeze(-1)
        maes.append(mae_metric(yhat, yb))
        yt = (yb >= threshold).any(dim=1).to(torch.int)
        yp = (yhat >= threshold).any(dim=1).to(torch.int)
        y_true_bin.extend(yt.cpu().tolist())
        y_pred_bin.extend(yp.cpu().tolist())
    mean_mae = sum(maes) / max(len(maes), 1)
    support0 = y_true_bin.count(0)
    support1 = y_true_bin.count(1)
    report_text = ''
    try:
        from sklearn.metrics import classification_report  # type: ignore
        report_text = classification_report(
            y_true_bin, y_pred_bin,
            target_names=["neg", "pos"],
            digits=4,
            zero_division=0,
        )
    except Exception:
        tp = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true_bin, y_pred_bin) if yt == 1 and yp == 0)
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0.0
        report_text = (
            f'Fallback report (sklearn not available)\n'
            f' support neg={support0} pos={support1}\n'
            f' TP={tp} TN={tn} FP={fp} FN={fn}  Acc={acc:.4f}'
        )
    return mean_mae, report_text, support0, support1


def build_dataset(dataset: int, history: int, horizon: int, stride: int, trunc_temp: float | None,
                  threshold: float,
                  enable_physics: bool = False,
                  data3_path: Path | None = None,
                  six_root: Path | None = None,
                  rebuild_cache: bool = False,
                  neighbor_seed: int = 42) -> Tuple[List, List, List, List, int, int, int]:
    """
    Returns X, Y, N, E, d_in, target_idx
    For dataset==3: channels (HD1,HD2,HD3), target HD1
    For dataset==6: fixed (target HD2, neighbors [HD3,HD1]) so d_in=3
    """
    if dataset == 3:
        # Defaults
        if data3_path is None:
            data3_path = Path('DataForFlashover/three-compartments/P-Flash_ThreeCompartments_all data 2020-6-26.xlsx')
        # caching
        cache_dir = data3_path.parent
        channels = ('HD1', 'HD2', 'HD3')
        target = 'HD1'
        # For physics in 3-comp: use neighbor=HD3, env=HD2 so N has 2 cols (neighbors then env)
        future_chs = ('HD3', 'HD2') if enable_physics else ()
        cache_name = (
            f"cross_cache3_h{history}_H{horizon}_s{stride}_t{trunc_temp if trunc_temp is not None else 'None'}_"
            f"phys{int(bool(enable_physics))}_fch{'+'.join(future_chs) if future_chs else 'None'}.pt"
        )
        cache_path = cache_dir / cache_name
        if cache_path.exists() and not rebuild_cache:
            logging.info(f'Loading cross-eval cache (3-comp): {cache_path}')
            import torch as _torch
            cached = _torch.load(cache_path)
            X, Y, E, N = cached['X'], cached['Y'], cached['E'], cached['N']
        else:
            series = load_timeseries(str(data3_path))
            X, Y, E, N = build_window_samples(
                series=series,
                history=history,
                horizon=horizon,
                stride=stride,
                use_channels=channels,
                target_channel=target,
                trunc_temp=trunc_temp,
                drop_after_fail=False,
                stop_when_all_channels_reach_trunc=True if trunc_temp is not None else False,
                future_channels=future_chs,
            )
            try:
                import torch as _torch
                _torch.save({'X': X, 'Y': Y, 'E': E, 'N': N,
                             'meta': {
                                 'dataset': 3, 'history': history, 'horizon': horizon, 'stride': stride,
                                 'trunc_temp': trunc_temp, 'enable_physics': bool(enable_physics),
                                 'future_channels': list(future_chs),
                             }}, cache_path)
                logging.info(f'Saved cross-eval cache (3-comp) to {cache_path}')
            except Exception as e:
                logging.warning(f'Failed to save cross-eval cache (3-comp): {e}')
        d_in = len(channels)
        target_idx = channels.index(target)
        # n_neighbors for physics: 1 (only HD3); env handled separately via N last column
        n_neighbors = 1 if enable_physics else 0
        return X, Y, N, E, d_in, target_idx, n_neighbors
    else:
        if six_root is None:
            six_root = Path('DataForFlashover/six-compartments')
        cache_dir = Path.cwd() / six_root
        cache_name = (
            f"cross_cache6_h{history}_H{horizon}_s{stride}_t{trunc_temp if trunc_temp is not None else 'None'}_"
            f"dynTargets_rand2_nei_seed{neighbor_seed}_phys{int(bool(enable_physics))}.pt"
        )
        cache_path = cache_dir / cache_name
        if cache_path.exists() and not rebuild_cache:
            logging.info(f'Loading cross-eval cache (6-comp): {cache_path}')
            import torch as _torch
            cached = _torch.load(cache_path)
            X, Y, E, N = cached['X'], cached['Y'], cached['E'], cached['N']
        else:
            series, exp_to_room, room_to_hd = load_six_series(str(Path.cwd() / six_root))
            # Dynamic mapping per experiment: target = fire room HD; neighbors = random 2 from remaining HDs
            code_to_name = {'1': 'Dining Room', '2': 'Kitchen', '4': 'Living Room', '5': 'Bedroom 1', '7': 'Bedroom 2', '8': 'Bedroom 3'}
            all_hds = ['HD1', 'HD2', 'HD3', 'HD4', 'HD5', 'HD6']
            rng = random.Random(neighbor_seed)
            exp_to_target = {}
            exp_to_neighbors = {}
            for exp in series.keys():
                room_num = exp_to_room.get(exp)
                name = code_to_name.get(room_num, None)
                if name and name in room_to_hd:
                    tgt = room_to_hd[name]
                else:
                    tgt = 'HD1'
                exp_to_target[exp] = tgt
                candidates = [h for h in all_hds if h != tgt]
                # deterministic per exp selection
                rng_exp = random.Random(neighbor_seed + int(exp))
                rng_exp.shuffle(candidates)
                exp_to_neighbors[exp] = candidates[:2]
            X, Y, E, N = build_window_samples_dynamic(
                series=series,
                exp_to_target_hd=exp_to_target,
                exp_to_neighbors=exp_to_neighbors,
                history=history,
                horizon=horizon,
                stride=stride,
                trunc_temp=trunc_temp,
                stop_when_all_channels_reach_trunc=True if trunc_temp is not None else False,
            )
            try:
                import torch as _torch
                _torch.save({'X': X, 'Y': Y, 'E': E, 'N': N,
                             'meta': {
                                 'dataset': 6, 'history': history, 'horizon': horizon, 'stride': stride,
                                 'trunc_temp': trunc_temp, 'enable_physics': bool(enable_physics),
                                 'neighbor_seed': neighbor_seed, 'neighbors_policy': 'rand2_except_target'
                             }}, cache_path)
                logging.info(f'Saved cross-eval cache (6-comp) to {cache_path}')
            except Exception as e:
                logging.warning(f'Failed to save cross-eval cache (6-comp): {e}')
        d_in = 3
        # In dynamic setup, input ordering is [nei1, nei2, target] so target index is 2
        target_idx = 2
        n_neighbors = 2 if enable_physics else 0
        return X, Y, N, E, d_in, target_idx, n_neighbors


def balance_indices(indices: List[int], Y: List[List[float]], threshold: float, seed: int) -> List[int]:
    pos_ids, neg_ids = [], []
    for idx in indices:
        is_pos = any(v >= threshold for v in Y[idx])
        (pos_ids if is_pos else neg_ids).append(idx)
    if not pos_ids or not neg_ids:
        return indices
    m = min(len(pos_ids), len(neg_ids))
    rng = random.Random(seed)
    rng.shuffle(pos_ids); rng.shuffle(neg_ids)
    out = pos_ids[:m] + neg_ids[:m]
    rng.shuffle(out)
    return out


def main():
    p = argparse.ArgumentParser(description='Cross-dataset evaluation (train on A, eval on B)')
    p.add_argument('--train_dataset', type=int, choices=[3, 6], required=True)
    p.add_argument('--eval_dataset', type=int, choices=[3, 6], required=True)
    p.add_argument('--history', type=int, default=96)
    p.add_argument('--horizon', type=int, default=120)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--trunc_temp', type=float, default=None)
    p.add_argument('--threshold', type=float, default=600.0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--model', type=str, default='LSTM',
                  help='mamba | RNN | LSTM | GRU | Bi-RNN | Bi-LSTM | Bi-GRU | Bi-LSTM-Attention | Bi-GRU-Attention')
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--rnn_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=1)
    p.add_argument('--d_state', type=int, default=64)
    p.add_argument('--d_conv', type=int, default=4)
    p.add_argument('--expand', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_balance', action='store_true')
    p.add_argument('--test_balance', action='store_true')
    p.add_argument('--enable_fp32_high_precision', action='store_true')
    # physics options for mamba
    p.add_argument('--enable_physics_loss', action='store_true')
    p.add_argument('--lambda_phys', type=float, default=0.0)
    p.add_argument('--dt', type=float, default=20.0)
    p.add_argument('--rebuild_cache', action='store_true')
    args = p.parse_args()

    assert args.train_dataset in (3, 6) and args.eval_dataset in (3, 6), 'dataset must be 3 or 6'

    set_seed(args.seed)
    device = get_device()

    # logging
    log_dir = Path('log'); log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = log_dir / f"cross_{args.model}_{args.train_dataset}to{args.eval_dataset}_{ts}.log"
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
    logging.info(f'Logging to {log_file}')

    if args.enable_fp32_high_precision:
        try:
            torch.set_float32_matmul_precision('high')
        except Exception as e:
            logging.warning(f'Failed to enable FP32 high precision matmul: {e}')

    # Build train dataset (dataset A)
    Xtr, Ytr, Ntr, Etr, din_train, tgt_idx_train, nnb_train = build_dataset(
        args.train_dataset, args.history, args.horizon, args.stride, args.trunc_temp, args.threshold,
        enable_physics=args.enable_physics_loss, rebuild_cache=args.rebuild_cache)
    folds_tr = make_folds(Etr, n_splits=5, seed=args.seed)
    train_idx, test_idx_tr = split_by_exp(Etr, folds_tr, 0)
    if args.train_balance:
        train_idx = balance_indices(train_idx, Ytr, args.threshold, args.seed)
    if args.test_balance:
        test_idx_tr = balance_indices(test_idx_tr, Ytr, args.threshold, args.seed+1)
    train_ds = WindowDataset(Xtr, Ytr, Ntr, indices=train_idx, clip_max=args.trunc_temp)
    test_ds_tr = WindowDataset(Xtr, Ytr, Ntr, indices=test_idx_tr, clip_max=args.trunc_temp)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader_tr = torch.utils.data.DataLoader(test_ds_tr, batch_size=args.batch_size, shuffle=False)

    # Build eval dataset (dataset B)
    Xev, Yev, Nev, Eev, din_eval, tgt_idx_eval, nnb_eval = build_dataset(
        args.eval_dataset, args.history, args.horizon, args.stride, args.trunc_temp, args.threshold,
        enable_physics=args.enable_physics_loss, rebuild_cache=args.rebuild_cache)
    folds_ev = make_folds(Eev, n_splits=5, seed=args.seed)
    _, test_idx_ev = split_by_exp(Eev, folds_ev, 0)  # only use first fold's test part
    if args.test_balance:
        test_idx_ev = balance_indices(test_idx_ev, Yev, args.threshold, args.seed+2)
    test_ds_ev = WindowDataset(Xev, Yev, Nev, indices=test_idx_ev, clip_max=args.trunc_temp)
    test_loader_ev = torch.utils.data.DataLoader(test_ds_ev, batch_size=args.batch_size, shuffle=False)

    # Ensure d_in matches
    if din_train != din_eval:
        logging.error(f'd_in mismatch between train ({din_train}) and eval ({din_eval}). Aborting.')
        return

    # Build model for training
    if args.model.lower() == 'mamba':
        model = MambaForecaster(
            d_in=din_train,
            d_model=args.d_model,
            n_layers=args.n_layers,
            horizon=args.horizon,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            d_out=1,
            enable_physics=args.enable_physics_loss,
            n_neighbors=nnb_train,
            lambda_phys=args.lambda_phys,
            dt=args.dt,
        ).to(device)
        def train_one_epoch():
            return model.train_one_epoch(train_loader, opt, crit, tgt_idx_train)
    else:
        rnn_type, bidir, use_attn = resolve_baseline_from_name(args.model)
        model = BaselineForecaster(
            d_in=din_train, horizon=args.horizon, rnn_type=rnn_type,
            hidden_size=args.hidden_size, num_layers=args.rnn_layers,
            bidirectional=bidir, use_attention=use_attn, dropout=args.dropout,
        ).to(device)
        def train_one_epoch():
            return model.train_one_epoch(train_loader, opt, crit, None)

    crit = nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mae = float('inf'); best_state = None; best_ep = 0; best_rep = ''
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch()
        val_mae, rep, s0, s1 = evaluate_model(model, test_loader_tr, device, args.threshold)
        improved = val_mae < best_mae
        if improved:
            best_mae = val_mae; best_ep = ep; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}; best_rep = rep
        logging.info(f'[TrainFold][Epoch {ep}/{args.epochs}] train L1: {tr_loss:.4f}  val MAE: {val_mae:.4f}' + ('  [improved]' if improved else ''))
        if improved:
            logging.info(f'Validation report (best so far):\n{rep}')

    logging.info(f'Best epoch: {best_ep}  best MAE: {best_mae:.4f}')

    # Rebuild a fresh model for eval dataset with same architecture, load best_state
    if args.model.lower() == 'mamba':
        eval_model = MambaForecaster(
            d_in=din_eval, d_model=args.d_model, n_layers=args.n_layers, horizon=args.horizon,
            d_state=args.d_state, d_conv=args.d_conv, expand=args.expand, d_out=1,
            enable_physics=False, n_neighbors=0, lambda_phys=0.0, dt=args.dt,
        ).to(device)
    else:
        rnn_type, bidir, use_attn = resolve_baseline_from_name(args.model)
        eval_model = BaselineForecaster(
            d_in=din_eval, horizon=args.horizon, rnn_type=rnn_type,
            hidden_size=args.hidden_size, num_layers=args.rnn_layers,
            bidirectional=bidir, use_attention=use_attn, dropout=args.dropout,
        ).to(device)
    if best_state is None:
        logging.error('No best state captured; aborting eval on target dataset')
        return
    # Load best_state; allow strict=False to ignore physics params mismatch across datasets
    missing, unexpected = [], []
    try:
        res = eval_model.load_state_dict(best_state, strict=False)
        missing = getattr(res, 'missing_keys', [])
        unexpected = getattr(res, 'unexpected_keys', [])
    except Exception as e:
        logging.error(f'Failed to load state dict into eval model: {e}')
        return
    if missing or unexpected:
        logging.info(f'Partial load: missing={missing} unexpected={unexpected}')
    mae_ev, rep_ev, s0_ev, s1_ev = evaluate_model(eval_model, test_loader_ev, device, args.threshold)
    logging.info('========== Cross-eval on target dataset =========')
    logging.info(f'MAE: {mae_ev:.4f}  Support: neg={s0_ev} pos={s1_ev}')
    logging.info(f'Classification report:\n{rep_ev}')


if __name__ == '__main__':
    main()
