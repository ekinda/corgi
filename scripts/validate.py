import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from corgi.config_corgiplus import config_corgiplus
from corgi.model import Corgi, CorgiPlus, CorgiPlusNofilm
from corgi.data_classes import CorgiDistributedSampler
from corgi.trainer_corgiplus import CorgiPlusDataset
from corgi.utils import load_experiment_mask, poisson_multinomial_masked_v2

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def _parse_bed_folds(bed_path: str):
    bed = np.loadtxt(bed_path, dtype=str)
    valid_seq = list(np.where(bed[:, 3] == 'fold4')[0])
    return valid_seq

def _load_ids(path: str) -> List[int]:
    with open(path) as f:
        return [int(x) for x in f.read().strip().split() if x.strip()]

def _init_dist(training_mode: str):
    if training_mode == 'local':
        return 0, 1, 0
    rank = int(torch.distributed.get_rank()) if dist.is_initialized() else int(np.getenv('SLURM_PROCID', 0))
    world = int(np.getenv('WORLD_SIZE', 1))
    return rank, world, rank % max(1, torch.cuda.device_count())

def _maybe_init_process_group(training_mode: str):
    if training_mode == 'local':
        return
    if dist.is_initialized():
        return
    rank = int(np.getenv('SLURM_PROCID', 0))
    world = int(np.getenv('WORLD_SIZE', 1))
    dist.init_process_group('nccl', rank=rank, world_size=world)

def _center_crop(label: torch.Tensor, target_len: int) -> torch.Tensor:
    curr = label.shape[-1]
    if curr == target_len:
        return label
    offset = (curr - target_len) // 2
    return label[..., offset : offset + target_len]

def _build_model(cfg: dict, mode: str, dnase_global_dim: int):
    if mode == 'corgi':
        return Corgi(cfg)
    if mode == 'corgiplus_rna' or mode == 'corgiplus_dnase':
        return CorgiPlus(cfg)
    if mode == 'corgiplus_dnase_nofilm':
        return CorgiPlusNofilm(cfg)
    raise ValueError(f"Unsupported mode {mode}")

def _mask_channels(exp_mask: torch.Tensor, keep: Sequence[int]) -> torch.Tensor:
    mask = exp_mask.clone()
    all_idx = set(range(exp_mask.shape[1]))
    drop = all_idx.difference(set(keep))
    if drop:
        drop_idx = torch.tensor(sorted(list(drop)), device=exp_mask.device)
        mask[:, drop_idx, :] = 0
    return mask


def _spearmanr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Spearman correlation on CPU using simple rank approximation (ties -> dense rank)."""
    if x.numel() < 2:
        return float('nan')
    x = x.float()
    y = y.float()

    def _rank(v: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(v)
        ranks = torch.empty_like(order, dtype=torch.float)
        ranks[order] = torch.arange(1, len(v) + 1, dtype=torch.float)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = torch.sqrt((vx ** 2).sum() * (vy ** 2).sum())
    if denom == 0:
        return float('nan')
    return float(torch.dot(vx, vy) / denom)


def _load_channel_names(path: str, output_channels: int) -> List[str]:
    names = [f"ch {i}" for i in range(output_channels)]
    try:
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for i in range(min(output_channels, len(lines))):
            # Use first column if tab-separated else full line
            parts = lines[i].split('\t')
            names[i] = parts[0]
    except Exception as e:
        logging.warning(f"Could not load channel names from {path}: {e}")
    return names

def evaluate(cfg: dict, mode: str, checkpoints: List[str], channels: Sequence[int], num_seqs: int, training_mode: str, max_tissues: int = None):
    _maybe_init_process_group(training_mode)
    rank, world_size, local_rank = _init_dist(training_mode)
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None

    exp_mask = load_experiment_mask(cfg['mask_path'])
    availability = {int(k): np.array(v) == 1 for k, v in exp_mask.items()}
    trans_reg = torch.from_numpy(np.load(cfg['trans_regulator_expression_path'])).float()
    channel_names = _load_channel_names(cfg.get('experiments_path', ''), cfg['output_channels'])
    valid_tissues = _load_ids(cfg['validation_tissues_path'])
    if max_tissues is not None and max_tissues > 0:
        rng = np.random.default_rng(cfg.get('seed', 1))
        subset_size = min(max_tissues, len(valid_tissues))
        valid_tissues = list(rng.choice(valid_tissues, size=subset_size, replace=False))
        valid_tissues.sort()
        if rank == 0:
            logging.info(f"Limiting validation to {len(valid_tissues)} tissues: {valid_tissues}")
    valid_seq = _parse_bed_folds(cfg['bed_path'])
    if num_seqs is not None:
        valid_seq = valid_seq[:num_seqs]

    dnase_global = None
    dnase_global_dim = cfg['input_trans_regulators']
    cfg = dict(cfg)

    if mode in ('corgiplus_dnase', 'corgiplus_dnase_nofilm'):
        dnase_global = torch.from_numpy(np.load(cfg['dnase_global_path'])).float().to(device)
        dnase_global_dim = dnase_global.shape[1]
        cfg['input_trans_regulators'] = dnase_global_dim
        cfg['corgiplus_aux_input_dim'] = 1
    elif mode == 'corgiplus_rna':
        cfg['corgiplus_aux_input_dim'] = 2

    eval_dataset = CorgiPlusDataset(
        dna_sequences=cfg['dna_path'],
        sequence_ids=valid_seq,
        tissue_dir=cfg['tissue_dir'],
        tissue_ids=valid_tissues,
        experiment_mask=exp_mask,
        trans_reg_expression=trans_reg,
        output_channels=cfg['output_channels'],
        augment_dna=False,
        augment_gnomad=False,
        augment_trans_reg_std=0.0,
        gnomad_pickle=None,
        trans_reg_clip=None,
        return_seq_id=False,
    )
    sampler = CorgiDistributedSampler(
        sequence_ids=valid_seq,
        tissue_ids=valid_tissues,
        num_processes=world_size,
        rank=rank,
        seed=cfg['seed'],
        shuffled=False,
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=cfg['batch_size'],
        sampler=sampler,
        num_workers=2,
        pin_memory=False,
    )
    model = _build_model(cfg, mode, dnase_global_dim).to(device)
    model.eval()

    for ckpt in checkpoints:
        state = torch.load(ckpt, map_location='cpu')
        state_dict = state.get('model_state_dict', state)
        model.load_state_dict(state_dict, strict=False)

        loss_sum = torch.zeros(1, device=device)
        count_sum = torch.zeros(1, device=device)

        sum_pred = torch.zeros(cfg['output_channels'], device=device)
        sum_label = torch.zeros(cfg['output_channels'], device=device)
        sum_pred2 = torch.zeros(cfg['output_channels'], device=device)
        sum_label2 = torch.zeros(cfg['output_channels'], device=device)
        sum_pred_label = torch.zeros(cfg['output_channels'], device=device)
        n_points = torch.zeros(cfg['output_channels'], device=device)

        tissue_stats: Dict[int, Dict[str, torch.Tensor]] = {}
        spearman_means = defaultdict(lambda: defaultdict(lambda: {'pred': [], 'label': []}))
        global_spearman = defaultdict(lambda: {'pred': [], 'label': []})

        with torch.no_grad():
            for batch in loader:
                dna_seq, trans_reg_b, label, exp_mask_b, tissue_id = batch

                dna_seq = dna_seq.to(device, non_blocking=True)
                trans_reg_b = trans_reg_b.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                exp_mask_b = exp_mask_b.to(device, non_blocking=True)

                exp_mask_sel = _mask_channels(exp_mask_b, channels)

                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if mode == 'corgiplus_rna':
                        aux_full = label[:, 16:18, :]
                        exp_mask_sel[:, 16:18, :] = 0
                        pred = model(dna_seq, aux_full.permute(0, 2, 1), trans_reg_b)
                    elif mode == 'corgiplus_dnase':
                        aux_full = label[:, 0:1, :]
                        exp_mask_sel[:, 0:1, :] = 0
                        cond = dnase_global[tissue_id.to(device)]
                        pred = model(dna_seq, aux_full.permute(0, 2, 1), cond)
                    elif mode == 'corgiplus_dnase_nofilm':
                        aux_full = label[:, 0:1, :]
                        exp_mask_sel[:, 0:1, :] = 0
                        pred = model(dna_seq, aux_full.permute(0, 2, 1), trans_reg=None)
                    elif mode == 'corgi':
                        pred = model(dna_seq, trans_reg_b)
                    else:
                        raise ValueError(mode)

                label_crop = _center_crop(label, cfg['output_central_bins'])
                pred_crop = _center_crop(pred, cfg['output_central_bins'])

                loss = poisson_multinomial_masked_v2(pred_crop, label_crop, exp_mask_sel, cfg['poisson_loss_weighting'], cfg['loss_epsilon'])
                loss = (loss * exp_mask_sel.squeeze(-1)).sum()
                denom = exp_mask_sel.sum()
                loss_sum += loss
                count_sum += denom

                mask_flat = exp_mask_sel.squeeze(-1).unsqueeze(-1)  # (B, C, 1)
                pred_flat = pred_crop
                label_flat = label_crop

                masked_pred = pred_flat * mask_flat
                masked_label = label_flat * mask_flat

                sum_pred += masked_pred.sum(dim=(0, 2))
                sum_label += masked_label.sum(dim=(0, 2))
                sum_pred2 += (masked_pred ** 2).sum(dim=(0, 2))
                sum_label2 += (masked_label ** 2).sum(dim=(0, 2))
                sum_pred_label += (masked_pred * masked_label).sum(dim=(0, 2))
                length = pred_flat.shape[2]
                n_points += mask_flat.sum(dim=(0, 2)) * length

                # Per-tissue Pearson accumulators
                unique_tissues = tissue_id.unique()
                for tid in unique_tissues:
                    tid_int = int(tid.item())
                    if tid_int not in tissue_stats:
                        tissue_stats[tid_int] = {
                            'sum_pred': torch.zeros(cfg['output_channels'], device=device),
                            'sum_label': torch.zeros(cfg['output_channels'], device=device),
                            'sum_pred2': torch.zeros(cfg['output_channels'], device=device),
                            'sum_label2': torch.zeros(cfg['output_channels'], device=device),
                            'sum_pred_label': torch.zeros(cfg['output_channels'], device=device),
                            'n_points': torch.zeros(cfg['output_channels'], device=device),
                        }
                    idx = (tissue_id == tid)
                    mp = masked_pred[idx]
                    ml = masked_label[idx]
                    ts = tissue_stats[tid_int]
                    ts['sum_pred'] += mp.sum(dim=(0, 2))
                    ts['sum_label'] += ml.sum(dim=(0, 2))
                    ts['sum_pred2'] += (mp ** 2).sum(dim=(0, 2))
                    ts['sum_label2'] += (ml ** 2).sum(dim=(0, 2))
                    ts['sum_pred_label'] += (mp * ml).sum(dim=(0, 2))
                    ts['n_points'] += mask_flat[idx].sum(dim=(0, 2)) * length

                    # Spearman on per-sequence means (keeps memory small)
                    pred_mean = pred_flat[idx].mean(dim=2)
                    label_mean = label_crop[idx].mean(dim=2)
                    mask_mean = exp_mask_sel[idx].squeeze(-1) > 0
                    if mask_mean.any():
                        avail = availability.get(tid_int, np.ones(mask_mean.shape[1], dtype=bool))
                        for ch in channels:
                            if ch >= mask_mean.shape[1] or ch >= len(avail) or not avail[ch] or not mask_mean[:, ch].any():
                                continue
                            pm = pred_mean[:, ch][mask_mean[:, ch]].detach().cpu()
                            lm = label_mean[:, ch][mask_mean[:, ch]].detach().cpu()
                            spearman_means[tid_int][ch]['pred'].append(pm)
                            spearman_means[tid_int][ch]['label'].append(lm)
                            global_spearman[ch]['pred'].append(pm)
                            global_spearman[ch]['label'].append(lm)

        if training_mode == 'slurm' and dist.is_initialized():
            for t in [loss_sum, count_sum, sum_pred, sum_label, sum_pred2, sum_label2, sum_pred_label, n_points]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        def _pearson_from_stats(sums):
            vals = {}
            for ch in channels:
                n = sums['n_points'][ch]
                if n.item() == 0:
                    vals[ch] = float('nan')
                    continue
                num = sums['sum_pred_label'][ch] - (sums['sum_pred'][ch] * sums['sum_label'][ch] / n)
                den = torch.sqrt((sums['sum_pred2'][ch] - (sums['sum_pred'][ch] ** 2) / n) * (sums['sum_label2'][ch] - (sums['sum_label'][ch] ** 2) / n))
                vals[ch] = float((num / (den + 1e-8)).item())
            return vals

        # Global metrics
        if rank == 0:
            mean_loss = (loss_sum / (count_sum + 1e-8)).item()
            logging.info(f"Checkpoint: {ckpt}")
            logging.info(f"Mean loss: {mean_loss:.4f}")
            logging.info("Per-tissue correlations (mean over requested channels, only available in mask):")

            pearson_ch_lists = defaultdict(list)
            spearman_ch_lists = defaultdict(list)
            for tid, stats in sorted(tissue_stats.items()):
                avail = availability.get(tid, np.ones(cfg['output_channels'], dtype=bool))
                pearson_t = _pearson_from_stats(stats)
                spearman_t = {}
                for ch in channels:
                    if ch >= len(avail) or not avail[ch]:
                        spearman_t[ch] = float('nan')
                        continue
                    pred_list = spearman_means[tid][ch]['pred']
                    label_list = spearman_means[tid][ch]['label']
                    if len(pred_list) == 0:
                        spearman_t[ch] = float('nan')
                        continue
                    pred_cat = torch.cat(pred_list)
                    label_cat = torch.cat(label_list)
                    spearman_t[ch] = _spearmanr_torch(pred_cat, label_cat)

                pearson_vals = [v for ch, v in pearson_t.items() if ch in channels and ch < len(avail) and avail[ch] and not np.isnan(v)]
                spearman_vals = [v for ch, v in spearman_t.items() if ch in channels and ch < len(avail) and avail[ch] and not np.isnan(v)]
                pearson_mean = float(np.nanmean(pearson_vals)) if pearson_vals else float('nan')
                spearman_mean = float(np.nanmean(spearman_vals)) if spearman_vals else float('nan')
                logging.info(f"  Tissue {tid}: Pearson_mean={pearson_mean:.4f}, Spearman_mean={spearman_mean:.4f}")

                for ch in channels:
                    if ch >= len(avail) or not avail[ch]:
                        continue
                    if not np.isnan(pearson_t.get(ch, float('nan'))):
                        pearson_ch_lists[ch].append(pearson_t[ch])
                    if not np.isnan(spearman_t.get(ch, float('nan'))):
                        spearman_ch_lists[ch].append(spearman_t[ch])

            logging.info("Global correlations by channel (tissue-mean):")
            pearson_means_global = []
            spearman_means_global = []
            for ch in channels:
                p_list = pearson_ch_lists.get(ch, [])
                s_list = spearman_ch_lists.get(ch, [])
                n = max(len(p_list), len(s_list))
                if n == 0:
                    continue
                p_mean = float(np.nanmean(p_list)) if p_list else float('nan')
                s_mean = float(np.nanmean(s_list)) if s_list else float('nan')
                pearson_means_global.append(p_mean)
                spearman_means_global.append(s_mean)
                logging.info(f"  ch {ch} ({channel_names[ch]}): Pearson={p_mean:.4f}, Spearman={s_mean:.4f}, n={n}")

            if pearson_means_global:
                pearson_mean_global = float(np.nanmean(pearson_means_global))
                spearman_mean_global = float(np.nanmean(spearman_means_global))
                logging.info(f"Global mean Pearson: {pearson_mean_global:.4f}, Global mean Spearman: {spearman_mean_global:.4f}")

    if training_mode == 'slurm' and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Corgi/CorgiPlus models')
    parser.add_argument('--mode', choices=['corgi', 'corgiplus_rna', 'corgiplus_dnase', 'corgiplus_dnase_nofilm'], required=True)
    parser.add_argument('--checkpoints', required=True, nargs='+', help='Checkpoint paths: list, wildcard-expanded by shell, or a text file containing one path per line')
    parser.add_argument('--channels', default='all', help='Comma-separated channel indices (0-21) to evaluate; default all')
    parser.add_argument('--num_seqs', type=int, default=None, help='Optional limit on number of validation sequences')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--training_mode', choices=['local', 'slurm'], default='local')
    parser.add_argument('--max_tissues', type=int, default=None, help='Optional limit on number of validation tissues (random subset)')
    args = parser.parse_args()

    cfg = dict(config_corgiplus)
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.batch_size:
        cfg['batch_size'] = args.batch_size

    if args.channels == 'all':
        channels = list(range(cfg['output_channels']))
    else:
        channels = [int(x) for x in args.channels.split(',') if x.strip()]

    # Flatten potential lists and handle txt file list
    raw_ckpts = []
    for entry in args.checkpoints:
        raw_ckpts.extend(entry.split(','))

    ckpts = []
    for entry in raw_ckpts:
        entry = entry.strip()
        if not entry:
            continue
        p = Path(entry)
        if p.is_file() and p.suffix == '.txt':
            with p.open() as f:
                ckpts.extend([line.strip() for line in f if line.strip()])
        else:
            ckpts.append(entry)

    print(f"Evaluating mode: {args.mode}")
    print(f"Checkpoints: {ckpts}")
    evaluate(cfg, args.mode, ckpts, channels, args.num_seqs, args.training_mode, args.max_tissues)
