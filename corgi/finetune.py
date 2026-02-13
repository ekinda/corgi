"""Fine-tuning helpers for adapting Corgi to new samples.

The finetuning entry-point expects raw genomics inputs that mirror the original
data generation pipeline:

* A reference genome FASTA and a BED file describing 524,288 bp regions.
* One BigWig per available genomic track (e.g. ``dnase``, ``h3k27ac``).
* A bulk RNA-seq profile provided as a :class:`pandas.Series` whose index lists
    gene symbols or Ensembl IDs.

CAUTION: This script is not tested yet.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import config_corgi
from .constants import BIN_SIZE, DEFAULT_TRACK_NAMES, WINDOW_SIZE
from .data_processing import build_bigwig_tensor, load_fasta_regions
from .predict import _load_model, _resolve_device
from .trans_regulators import encode_trans_reg_expression
from .utils import poisson_multinomial_masked_v2

N_TRACKS = len(DEFAULT_TRACK_NAMES)
N_BINS_FULL = WINDOW_SIZE // BIN_SIZE


def _crop_to_central(tensor: torch.Tensor, target_bins: int) -> torch.Tensor:
    current = tensor.shape[-1]
    if current == target_bins:
        return tensor
    pad = (current - target_bins) // 2
    return tensor[..., pad : pad + target_bins]


def _read_bed_regions(bed_path: Union[str, Path]) -> Sequence[Tuple[str, int, int]]:
    regions = []
    with Path(bed_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start_str, end_str, *_ = line.rstrip().split("\t")
            regions.append((chrom, int(start_str), int(end_str)))
    return regions


def _prepare_finetune_tensors(
    fasta_path: Union[str, Path],
    bed_path: Union[str, Path],
    bigwig_tracks: Mapping[str, Union[str, Path]],
    track_order: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    regions = _read_bed_regions(bed_path)
    sequences = load_fasta_regions(fasta_path, regions)
    normalised_map = _normalise_bigwig_tracks(bigwig_tracks, track_order)
    coverage, mask = build_bigwig_tensor(normalised_map, regions, track_order)
    dna_array = np.stack(sequences, axis=0).astype(np.float32, copy=False)
    return dna_array, coverage.astype(np.float32, copy=False), mask.astype(np.float32, copy=False)


def _normalise_bigwig_tracks(
    bigwig_tracks: Mapping[str, Union[str, Path]],
    track_order: Sequence[str],
) -> Mapping[str, Union[str, Path]]:
    order_lookup = {name.lower(): name for name in track_order}
    resolved = {}
    for name, path in bigwig_tracks.items():
        key = order_lookup.get(name.lower())
        if key is None:
            raise KeyError(f"Unrecognised track '{name}'. Expected one of {list(track_order)}")
        resolved[key] = path
    return resolved


class FinetuneDataset(Dataset):
    def __init__(
        self,
        dna_array: np.ndarray,
        coverage_array: np.ndarray,
        mask: np.ndarray,
        trans_reg_vector: np.ndarray,
    ) -> None:
        self.dna = torch.from_numpy(dna_array).to(dtype=torch.float32)
        self.coverage = torch.from_numpy(coverage_array).to(dtype=torch.float32)
        self.mask = torch.from_numpy(mask).to(dtype=torch.float32).unsqueeze(-1)
        self.trans_reg = torch.from_numpy(trans_reg_vector).to(dtype=torch.float32)

    def __len__(self) -> int:
        return self.dna.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.dna[idx],
            self.trans_reg.clone(),
            self.coverage[idx],
            self.mask.clone(),
        )


@dataclass
class FinetuneSettings:
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0
    dtype: torch.dtype = torch.bfloat16
    log_every: int = 50
    num_workers: int = 0


def finetune_corgi(
    checkpoint_path: Union[str, Path],
    fasta_path: Union[str, Path],
    bed_path: Union[str, Path],
    bigwig_tracks: Mapping[str, Union[str, Path]],
    rna_seq_expression: Union[pd.Series, np.ndarray, Sequence[float]],
    output_dir: Union[str, Path],
    *,
    settings: Optional[FinetuneSettings] = None,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[dict] = None,
    trainable_modules: Optional[Sequence[str]] = None,
) -> Path:
    cfg = dict(config_corgi)
    if config is not None:
        cfg.update(config)
    settings = settings or FinetuneSettings()

    dna_array, coverage_array, mask = _prepare_finetune_tensors(
        fasta_path,
        bed_path,
        bigwig_tracks,
        DEFAULT_TRACK_NAMES,
    )
    trans_reg_vector = encode_trans_reg_expression(rna_seq_expression, cfg)

    dataset = FinetuneDataset(dna_array, coverage_array, mask, trans_reg_vector)
    dataloader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=settings.num_workers,
    )

    resolved_device = _resolve_device(device)
    model = _load_model(checkpoint_path, cfg, resolved_device)

    if trainable_modules is not None:
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in trainable_modules)

    optim_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=settings.learning_rate, weight_decay=settings.weight_decay)

    for epoch in range(settings.epochs):
        for step, (dna_seq, trans_reg, label, track_mask) in enumerate(dataloader, start=1):
            dna_seq = dna_seq.to(resolved_device)
            trans_reg = trans_reg.to(resolved_device)
            label = label.to(resolved_device)
            track_mask = track_mask.to(resolved_device)

            optimizer.zero_grad()
            autocast_dtype = settings.dtype if resolved_device.type == "cuda" else torch.float32
            with torch.autocast(device_type=resolved_device.type, dtype=autocast_dtype):
                preds = model(dna_seq, trans_reg)
                cropped_label = _crop_to_central(label, cfg["output_central_bins"])
                channel_losses = poisson_multinomial_masked_v2(
                    preds,
                    cropped_label,
                    track_mask,
                    total_weight=cfg["poisson_loss_weighting"],
                    epsilon=cfg["loss_epsilon"],
                )
                loss = (channel_losses * track_mask.squeeze(-1)).sum() / track_mask.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, settings.gradient_clipping)
            optimizer.step()

            if settings.log_every and step % settings.log_every == 0:
                print(f"Epoch {epoch + 1} Step {step} Loss {loss.item():.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"corgi_finetuned_{timestamp}.pt"
    torch.save({"model_state_dict": model.state_dict()}, output_path)
    return output_path
