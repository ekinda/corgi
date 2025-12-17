import argparse
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from corgi.config_corgiplus import config_corgiplus
from corgi.model import Corgi, CorgiPlus, CorgiPlusNofilm
from corgi.data_classes import CorgiDataset, CorgiDistributedSampler
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

def evaluate(cfg: dict, mode: str, checkpoints: List[str], channels: Sequence[int], num_seqs: int, training_mode: str):
    _maybe_init_process_group(training_mode)
    rank, world_size, local_rank = _init_dist(training_mode)
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None

    exp_mask = load_experiment_mask(cfg['mask_path'])
    trans_reg = torch.from_numpy(np.load(cfg['trans_regulator_expression_path'])).float()
    valid_tissues = _load_ids(cfg['validation_tissues_path'])
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

    dataset_cls = CorgiPlusDataset if mode.startswith('corgiplus') else CorgiDataset
    eval_dataset = dataset_cls(
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

        with torch.no_grad():
            for batch in loader:
                if mode.startswith('corgiplus'):
                    dna_seq, trans_reg_b, label, exp_mask_b, tissue_id = batch
                else:
                    dna_seq, trans_reg_b, label, exp_mask_b = batch
                    tissue_id = None

                dna_seq = dna_seq.to(device, non_blocking=True)
                trans_reg_b = trans_reg_b.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                exp_mask_b = exp_mask_b.to(device, non_blocking=True)

                exp_mask_sel = _mask_channels(exp_mask_b, channels)
                label_crop = _center_crop(label, cfg['output_central_bins'])

                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if mode == 'corgiplus_rna':
                        aux = label_crop[:, 16:18, :]
                        exp_mask_sel[:, 16:18, :] = 0
                        pred = model(dna_seq, aux.permute(0, 2, 1), trans_reg_b)
                    elif mode == 'corgiplus_dnase':
                        aux = label_crop[:, 0:1, :]
                        exp_mask_sel[:, 0:1, :] = 0
                        cond = dnase_global[tissue_id.to(device)]
                        pred = model(dna_seq, aux.permute(0, 2, 1), cond)
                    elif mode == 'corgiplus_dnase_nofilm':
                        aux = label_crop[:, 0:1, :]
                        exp_mask_sel[:, 0:1, :] = 0
                        pred = model(dna_seq, aux.permute(0, 2, 1), trans_reg=None)
                    elif mode == 'corgi':
                        pred = model(dna_seq, trans_reg_b)
                    else:
                        raise ValueError(mode)

                loss = poisson_multinomial_masked_v2(pred, label_crop, exp_mask_sel, cfg['poisson_loss_weighting'], cfg['loss_epsilon'])
                loss = (loss * exp_mask_sel.squeeze(-1)).sum()
                denom = exp_mask_sel.sum()
                loss_sum += loss
                count_sum += denom

                mask_flat = exp_mask_sel.squeeze(-1).unsqueeze(-1)  # (B, C, 1)
                pred_flat = pred[:, :, :cfg['output_central_bins']]
                label_flat = label_crop[:, :, :cfg['output_central_bins']]

                masked_pred = pred_flat * mask_flat
                masked_label = label_flat * mask_flat

                sum_pred += masked_pred.sum(dim=(0, 2))
                sum_label += masked_label.sum(dim=(0, 2))
                sum_pred2 += (masked_pred ** 2).sum(dim=(0, 2))
                sum_label2 += (masked_label ** 2).sum(dim=(0, 2))
                sum_pred_label += (masked_pred * masked_label).sum(dim=(0, 2))
                n_points += mask_flat.sum(dim=(0, 2))

        if training_mode == 'slurm' and dist.is_initialized():
            for t in [loss_sum, count_sum, sum_pred, sum_label, sum_pred2, sum_label2, sum_pred_label, n_points]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        if rank == 0:
            mean_loss = (loss_sum / (count_sum + 1e-8)).item()
            pearson = []
            for ch in range(cfg['output_channels']):
                if ch not in channels or n_points[ch].item() == 0:
                    pearson.append(float('nan'))
                    continue
                n = n_points[ch]
                num = sum_pred_label[ch] - (sum_pred[ch] * sum_label[ch] / n)
                den = torch.sqrt((sum_pred2[ch] - (sum_pred[ch] ** 2) / n) * (sum_label2[ch] - (sum_label[ch] ** 2) / n))
                pearson.append((num / (den + 1e-8)).item())

            logging.info(f"Checkpoint: {ckpt}")
            logging.info(f"Mean loss: {mean_loss:.4f}")
            logging.info("Pearson by channel:")
            for ch in channels:
                logging.info(f"  ch {ch}: {pearson[ch]:.4f}")

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
    evaluate(cfg, args.mode, ckpts, channels, args.num_seqs, args.training_mode)
