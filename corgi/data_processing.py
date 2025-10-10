"""Data preparation utilities shared between training and fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pyBigWig
from pyfaidx import Fasta

from .constants import BIN_SIZE, DEFAULT_TRACK_NAMES, WINDOW_SIZE

TRANSFORM_PARAMS: Dict[str, Dict[str, float]] = {
    "dnase": {"clip": 64, "soft_clip": 32, "scale": 2.0, "sum_stat": "mean"},
    "atac": {"clip": 64, "soft_clip": 32, "scale": 1.0, "sum_stat": "mean"},
    "h3k4me1": {"clip": 64, "soft_clip": 48, "scale": 1.0, "sum_stat": "mean"},
    "h3k4me2": {"clip": 64, "soft_clip": 48, "scale": 1.0, "sum_stat": "mean"},
    "h3k4me3": {"clip": 64, "soft_clip": 48, "scale": 1.0, "sum_stat": "mean"},
    "h3k9ac": {"clip": 64, "soft_clip": 48, "scale": 1.0, "sum_stat": "mean"},
    "h3k9me3": {"clip": 64, "soft_clip": 48, "scale": 1.0, "sum_stat": "mean"},
    "h3k27ac": {"clip": 64, "soft_clip": 48, "scale": 1.0, "sum_stat": "mean"},
    "h3k27me3": {"clip": 64, "soft_clip": 48, "scale": 1.0, "sum_stat": "mean"},
    "h3k36me3": {"clip": 64, "soft_clip": 48, "scale": 1.0, "sum_stat": "mean"},
    "h3k79me2": {"clip": 64, "soft_clip": 48, "scale": 1.0, "sum_stat": "mean"},
    "ctcf": {"clip": 64, "soft_clip": 48, "scale": 1.0, "sum_stat": "mean"},
    "cage_plus": {"clip": 512, "soft_clip": 384, "scale": 1.0, "sum_stat": "sum"},
    "cage_minus": {"clip": 512, "soft_clip": 384, "scale": 1.0, "sum_stat": "sum"},
    "rampage_plus": {"clip": 512, "soft_clip": 384, "scale": 1.0, "sum_stat": "sum"},
    "rampage_minus": {"clip": 512, "soft_clip": 384, "scale": 1.0, "sum_stat": "sum"},
    "rna_total_plus": {"clip": 512, "soft_clip": 384, "scale": 1.0, "sum_stat": "sum_sqrt"},
    "rna_total_minus": {"clip": 512, "soft_clip": 384, "scale": 1.0, "sum_stat": "sum_sqrt"},
    "rna_polya_plus": {"clip": 512, "soft_clip": 384, "scale": 1.0, "sum_stat": "sum_sqrt"},
    "rna_polya_minus": {"clip": 512, "soft_clip": 384, "scale": 1.0, "sum_stat": "sum_sqrt"},
    "rna_10x": {"clip": 512, "soft_clip": 384, "scale": 1.0, "sum_stat": "sum_sqrt"},
    "wgbs": {"clip": 128, "soft_clip": 64, "scale": 1.0, "sum_stat": "mean"},
}


def process_coverage(values: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    coverage = values.astype(np.float32, copy=False)
    scale = params.get("scale", 1.0)
    soft_clip_val = params.get("soft_clip")
    clip_val = params.get("clip")

    coverage *= scale

    if soft_clip_val is not None:
        threshold = float(soft_clip_val)
        coverage = np.minimum(coverage, threshold + np.sqrt(np.maximum(0, coverage - threshold)))
    if clip_val is not None:
        coverage = np.clip(coverage, 0, clip_val)
    return coverage


def fast_bin(raw_coverage: np.ndarray, sum_stat: str) -> np.ndarray:
    reshaped = raw_coverage.reshape(-1, BIN_SIZE)
    if sum_stat == "mean":
        return reshaped.mean(axis=1)
    if sum_stat == "sum":
        return reshaped.sum(axis=1)
    if sum_stat == "sum_sqrt":
        return np.sqrt(reshaped.sum(axis=1))
    raise ValueError(f"Unsupported sum_stat: {sum_stat}")


def load_bigwig_bins(
    bigwig_path: Union[str, Path],
    chrom: str,
    start: int,
    end: int,
    track_name: str,
) -> np.ndarray:
    params = TRANSFORM_PARAMS.get(track_name.lower(), {})
    with pyBigWig.open(str(bigwig_path), "r") as bw:
        values = bw.values(chrom, start, end, numpy=True)
    if values is None:
        values = np.zeros(end - start, dtype=np.float32)
    coverage = np.asarray(values, dtype=np.float32)
    coverage[np.isnan(coverage)] = 0.0
    processed = process_coverage(coverage, params)
    binned = fast_bin(processed, params.get("sum_stat", "mean"))
    return binned.astype(np.float32)


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = sequence.upper()
    one_hot = np.zeros((len(seq), 4), dtype=np.float32)
    for idx, base in enumerate(seq):
        column = mapping.get(base, None)
        if column is not None:
            one_hot[idx, column] = 1.0
    return one_hot


def load_fasta_regions(fasta_path: Union[str, Path], regions: Sequence[Tuple[str, int, int]]) -> List[np.ndarray]:
    fasta = Fasta(str(fasta_path))
    sequences: List[np.ndarray] = []
    try:
        for chrom, start, end in regions:
            seq = fasta[chrom][start:end].seq
            if len(seq) != WINDOW_SIZE:
                raise ValueError(f"Region {chrom}:{start}-{end} length mismatch (expected {WINDOW_SIZE}).")
            sequences.append(one_hot_encode_sequence(seq))
    finally:
        fasta.close()
    return sequences


def build_bigwig_tensor(
    bigwig_map: Mapping[str, Union[str, Path]],
    regions: Sequence[Tuple[str, int, int]],
    track_order: Sequence[str] = DEFAULT_TRACK_NAMES,
) -> Tuple[np.ndarray, np.ndarray]:
    track_to_index = {name: idx for idx, name in enumerate(track_order)}
    mask = np.zeros(len(track_order), dtype=np.float32)
    tensors = []
    for chrom, start, end in regions:
        track_matrix = np.zeros((len(track_order), WINDOW_SIZE // BIN_SIZE), dtype=np.float32)
        for track_name, path in bigwig_map.items():
            if track_name not in track_to_index:
                raise KeyError(f"Unrecognised track '{track_name}'. Expected one of {list(track_order)}")
            idx = track_to_index[track_name]
            track_matrix[idx] = load_bigwig_bins(path, chrom, start, end, track_name)
            mask[idx] = 1.0
        tensors.append(track_matrix)
    stacked = np.stack(tensors, axis=0)
    return stacked, mask
