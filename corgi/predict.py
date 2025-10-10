"""High-level inference helpers for the Corgi model.

The functions in this module expose the three user-facing prediction entry
points described in the packaging plan:

1. `predict_sequence` runs Corgi on a single 524,288 bp DNA sequence using a
   provided trans regulator expression profile.
2. `predict_regions` iterates over genomic regions defined by a FASTA/BED pair
   and stacks the predictions for each 524,288 bp window.
3. `predict_regions_with_bigwig` mirrors (2) but also materialises one BigWig
   file per track for convenient downstream browsing.

All helpers share a common loading path that builds a model instance from a
checkpoint, applies the same configuration that was used during pretraining,
converts the user-provided trans regulator inputs into the fixed ordering that
Corgi expects, and handles batching logic for efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyBigWig
import torch
from pyfaidx import Fasta

from .config import config_corgi
from .constants import BIN_SIZE, CENTRAL_BINS, FLANK, STRIDE, WINDOW_SIZE, DEFAULT_TRACK_NAMES
from .model import Corgi
from .trans_regulators import encode_trans_reg_expression

_DNA_TO_INDEX = np.full(256, -1, dtype=np.int16)
_DNA_TO_INDEX[ord("A")] = 0
_DNA_TO_INDEX[ord("C")] = 1
_DNA_TO_INDEX[ord("G")] = 2
_DNA_TO_INDEX[ord("T")] = 3


@dataclass
class _Window:
    chrom: str
    start: int
    end: int

    @property
    def central_start(self) -> int:
        return self.start + FLANK

    @property
    def central_end(self) -> int:
        return self.end - FLANK


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_model(checkpoint_path: Union[str, Path], config: dict, device: torch.device) -> Corgi:
    model = Corgi(config)
    state = torch.load(checkpoint_path, map_location=device)
    weights = state.get("model_state_dict", state)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()
    return model


def _prepare_trans_regulator_vector(
    expression: Union[np.ndarray, Sequence[float], pd.Series],
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    ordered = encode_trans_reg_expression(expression, config)
    tensor = torch.from_numpy(ordered)
    return tensor.to(device).unsqueeze(0)


def _sequence_to_array(sequence: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(sequence, str):
        text = sequence.upper()
        if len(text) != WINDOW_SIZE:
            raise ValueError(f"DNA sequence must be {WINDOW_SIZE} bp long (got {len(text)}).")
        codes = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        indices = _DNA_TO_INDEX[codes]
        one_hot = np.zeros((len(text), 4), dtype=np.float32)
        valid = indices >= 0
        rows = np.where(valid)[0]
        one_hot[rows, indices[valid]] = 1.0
        return one_hot
    if isinstance(sequence, torch.Tensor):
        arr = sequence.detach().cpu().to(dtype=torch.float32).numpy()
    else:
        arr = np.asarray(sequence, dtype=np.float32)
    if arr.shape != (WINDOW_SIZE, 4):
        raise ValueError(f"Expected array of shape ({WINDOW_SIZE}, 4) (got {arr.shape}).")
    return arr


def _stack_sequences(batch: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(batch, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(array).to(device)


def _enumerate_regions(bed_path: Union[str, Path]) -> List[Tuple[str, int, int]]:
    regions: List[Tuple[str, int, int]] = []
    with Path(bed_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start_str, end_str, *_ = line.rstrip().split("\t")
            regions.append((chrom, int(start_str), int(end_str)))
    return regions


def _tile_region(start: int, end: int) -> Iterable[Tuple[int, int]]:
    if end - start < WINDOW_SIZE:
        raise ValueError("Regions shorter than 524,288 bp are not supported.")
    last_start = end - WINDOW_SIZE
    position = start
    while position + WINDOW_SIZE <= end:
        yield position, position + WINDOW_SIZE
        position += STRIDE
    if position - STRIDE != last_start:
        yield last_start, last_start + WINDOW_SIZE


def _predict_batch(
    model: Corgi,
    sequences: Sequence[np.ndarray],
    trans_reg_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    seq_tensor = _stack_sequences(sequences, device)
    batch_size = seq_tensor.size(0)
    trans_reg_batch = trans_reg_tensor.expand(batch_size, -1)
    with torch.no_grad():
        outputs = model(seq_tensor, trans_reg_batch)
    return outputs.permute(0, 2, 1).cpu().numpy()


def _write_bigwig(
    predictions: np.ndarray,
    windows: Sequence[_Window],
    fasta: Fasta,
    track_names: Sequence[str],
    output_dir: Union[str, Path],
    prefix: str,
) -> None:
    chrom_sizes = {name: entry.len for name, entry in fasta.faidx.items()}
    header = [(chrom, size) for chrom, size in chrom_sizes.items()]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = []
    for track_idx, track_name in enumerate(track_names):
        path = out_dir / f"{prefix}_{track_name}.bw"
        writer = pyBigWig.open(str(path), "w")
        writer.addHeader(header)
        writers.append(writer)
    try:
        for pred, window in zip(predictions, windows):
            starts = window.central_start + np.arange(CENTRAL_BINS) * BIN_SIZE
            ends = starts + BIN_SIZE
            chrom = window.chrom
            for track_idx, writer in enumerate(writers):
                writer.addEntries(
                    [chrom] * CENTRAL_BINS,
                    starts.tolist(),
                    ends.tolist(),
                    pred[:, track_idx].astype(float).tolist(),
                )
    finally:
        for writer in writers:
            writer.close()


def _load_components(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]],
    config: Optional[dict],
    track_names: Optional[Sequence[str]],
) -> Tuple[Corgi, torch.device, dict, Tuple[str, ...]]:
    cfg = dict(config_corgi)
    if config is not None:
        cfg.update(config)
    resolved_device = _resolve_device(device)
    model = _load_model(checkpoint_path, cfg, resolved_device)
    names = tuple(track_names) if track_names is not None else DEFAULT_TRACK_NAMES
    return model, resolved_device, cfg, names


def predict_sequence(
    checkpoint_path: Union[str, Path],
    sequence: Union[str, np.ndarray, torch.Tensor],
    trans_reg_expression: Union[np.ndarray, Sequence[float], pd.Series],
    *,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[dict] = None,
) -> np.ndarray:
    model, resolved_device, cfg, _ = _load_components(checkpoint_path, device, config, None)
    trans_reg_tensor = _prepare_trans_regulator_vector(trans_reg_expression, cfg, resolved_device)
    seq_array = _sequence_to_array(sequence)
    prediction = _predict_batch(model, [seq_array], trans_reg_tensor, resolved_device)
    return prediction[0]


def predict_regions(
    checkpoint_path: Union[str, Path],
    fasta_path: Union[str, Path],
    bed_path: Union[str, Path],
    trans_reg_expression: Union[np.ndarray, Sequence[float], pd.Series],
    *,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[dict] = None,
    batch_size: int = 2,
) -> np.ndarray:
    model, resolved_device, cfg, _ = _load_components(checkpoint_path, device, config, None)
    trans_reg_tensor = _prepare_trans_regulator_vector(trans_reg_expression, cfg, resolved_device)
    fasta = Fasta(str(fasta_path))
    regions = _enumerate_regions(bed_path)
    windows: List[_Window] = []
    sequences: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    for chrom, region_start, region_end in regions:
        for window_start, window_end in _tile_region(region_start, region_end):
            seq = fasta[chrom][window_start:window_end].seq
            sequences.append(_sequence_to_array(seq))
            windows.append(_Window(chrom, window_start, window_end))
            if len(sequences) == batch_size:
                predictions.append(_predict_batch(model, sequences, trans_reg_tensor, resolved_device))
                sequences.clear()
    if sequences:
        predictions.append(_predict_batch(model, sequences, trans_reg_tensor, resolved_device))
    if not predictions:
        raise ValueError("No prediction windows produced from the supplied BED file.")
    fasta.close()
    return np.concatenate(predictions, axis=0)


def predict_regions_with_bigwig(
    checkpoint_path: Union[str, Path],
    fasta_path: Union[str, Path],
    bed_path: Union[str, Path],
    trans_reg_expression: Union[np.ndarray, Sequence[float], pd.Series],
    *,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[dict] = None,
    batch_size: int = 2,
    bigwig_dir: Union[str, Path] = "corgi_bigwig",
    bigwig_prefix: str = "corgi",
) -> np.ndarray:
    model, resolved_device, cfg, track_names = _load_components(checkpoint_path, device, config, None)
    trans_reg_tensor = _prepare_trans_regulator_vector(trans_reg_expression, cfg, resolved_device)
    fasta = Fasta(str(fasta_path))
    regions = _enumerate_regions(bed_path)
    windows: List[_Window] = []
    sequences: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    for chrom, region_start, region_end in regions:
        for window_start, window_end in _tile_region(region_start, region_end):
            seq = fasta[chrom][window_start:window_end].seq
            sequences.append(_sequence_to_array(seq))
            windows.append(_Window(chrom, window_start, window_end))
            if len(sequences) == batch_size:
                predictions.append(_predict_batch(model, sequences, trans_reg_tensor, resolved_device))
                sequences.clear()
    if sequences:
        predictions.append(_predict_batch(model, sequences, trans_reg_tensor, resolved_device))
    if not predictions:
        raise ValueError("No prediction windows produced from the supplied BED file.")
    stacked = np.concatenate(predictions, axis=0)
    _write_bigwig(stacked, windows, fasta, track_names, bigwig_dir, bigwig_prefix)
    fasta.close()
    return stacked
