"""High-level inference helpers for pretrained Corgi-family models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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

_DEFAULT_TF_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "tf_reference.npy"
_TF_REFERENCE_CACHE: Dict[str, np.ndarray] = {}


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
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()
    return model


def quantile_normalize_trans_reg_expression(expression: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Quantile-normalize a trans-regulator vector to a reference distribution."""
    expr = np.asarray(expression, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    if expr.ndim != 1:
        raise ValueError(f"Expected 1D expression vector, received shape {expr.shape}.")
    if ref.ndim != 1:
        raise ValueError(f"Expected 1D reference vector, received shape {ref.shape}.")
    if expr.shape[0] != ref.shape[0]:
        raise ValueError(
            f"Expression/reference length mismatch: {expr.shape[0]} vs {ref.shape[0]}."
        )

    order = np.argsort(expr, kind="mergesort")
    ref_sorted = np.sort(ref)
    normalized = np.empty_like(expr, dtype=np.float32)
    normalized[order] = ref_sorted
    return normalized


def _load_tf_reference(config: dict) -> np.ndarray:
    reference_path = Path(config.get("tf_reference_path", _DEFAULT_TF_REFERENCE_PATH)).resolve()
    cache_key = str(reference_path)
    if cache_key not in _TF_REFERENCE_CACHE:
        if not reference_path.exists():
            raise FileNotFoundError(
                f"TF reference file not found at {reference_path}. "
                "Set config['tf_reference_path'] to override the default."
            )
        _TF_REFERENCE_CACHE[cache_key] = np.load(reference_path).astype(np.float32, copy=False)
    return _TF_REFERENCE_CACHE[cache_key]


def _prepare_trans_regulator_vector(
    expression: Union[np.ndarray, Sequence[float], pd.Series],
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    ordered = encode_trans_reg_expression(expression, config)
    reference = _load_tf_reference(config)
    normalized = quantile_normalize_trans_reg_expression(ordered, reference)
    tensor = torch.from_numpy(normalized)
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
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(seq_tensor, trans_reg_batch)
    return outputs.permute(0, 2, 1).cpu().numpy()


def _resolve_output_channels(
    output_channels: Optional[Sequence[Union[int, str]]],
    track_names: Sequence[str],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    if output_channels is None:
        indices = np.arange(len(track_names), dtype=np.int64)
        names = tuple(track_names)
        return indices, names

    indices: List[int] = []
    for channel in output_channels:
        if isinstance(channel, int):
            idx = channel
        else:
            if channel not in track_names:
                raise ValueError(f"Unknown output channel name '{channel}'.")
            idx = track_names.index(channel)
        if idx < 0 or idx >= len(track_names):
            raise ValueError(f"Output channel index {idx} is out of bounds for {len(track_names)} tracks.")
        if idx not in indices:
            indices.append(idx)

    selected = np.asarray(indices, dtype=np.int64)
    names = tuple(track_names[i] for i in selected)
    return selected, names


def _write_bigwig(
    predictions: np.ndarray,
    windows: Sequence[_Window],
    chrom_sizes_path: Union[str, Path],
    track_names: Sequence[str],
    output_dir: Union[str, Path],
    prefix: str,
) -> None:
    header: List[Tuple[str, int]] = []
    with Path(chrom_sizes_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, size_str, *_ = line.rstrip().split("\t")
            header.append((chrom, int(size_str)))
    if not header:
        raise ValueError(f"No chromosome entries found in chrom.sizes file: {chrom_sizes_path}")

    chrom_size_map: Dict[str, int] = dict(header)
    chrom_rank: Dict[str, int] = {chrom: i for i, (chrom, _) in enumerate(header)}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = []
    for track_idx, track_name in enumerate(track_names):
        path = out_dir / f"{prefix}_{track_name}.bw"
        writer = pyBigWig.open(str(path), "w")
        writer.addHeader(header)
        writers.append(writer)

    write_order = sorted(
        range(len(windows)),
        key=lambda i: (chrom_rank.get(windows[i].chrom, len(chrom_rank)), windows[i].central_start),
    )

    last_start_by_writer: List[Dict[str, int]] = [{chrom: -1 for chrom in chrom_size_map} for _ in writers]
    try:
        for idx in write_order:
            pred = predictions[idx]
            window = windows[idx]
            starts = window.central_start + np.arange(CENTRAL_BINS) * BIN_SIZE
            ends = starts + BIN_SIZE
            chrom = window.chrom
            chrom_size = chrom_size_map.get(chrom)
            if chrom_size is None:
                continue

            valid = (starts >= 0) & (ends <= chrom_size) & (starts < ends)
            if not np.any(valid):
                continue

            starts_v = starts[valid]
            ends_v = ends[valid]

            for track_idx, writer in enumerate(writers):
                values_v = pred[valid, track_idx].astype(float)
                last_start = last_start_by_writer[track_idx][chrom]
                increasing = starts_v > last_start
                if not np.any(increasing):
                    continue

                starts_i = starts_v[increasing]
                ends_i = ends_v[increasing]
                values_i = values_v[increasing]
                writer.addEntries(
                    [chrom] * len(starts_i),
                    starts_i.tolist(),
                    ends_i.tolist(),
                    values_i.tolist(),
                )
                last_start_by_writer[track_idx][chrom] = int(starts_i[-1])
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


class corgi_pretrained:
    """Inference wrapper for pretrained Corgi checkpoints."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[dict] = None,
        output_channels: Optional[Sequence[Union[int, str]]] = None,
    ):
        self.model, self.device, self.config, all_track_names = _load_components(
            checkpoint_path,
            device,
            config,
            DEFAULT_TRACK_NAMES,
        )
        self.output_channel_indices, self.output_track_names = _resolve_output_channels(output_channels, all_track_names)

    def _select_channels(self, prediction: np.ndarray) -> np.ndarray:
        return prediction[:, self.output_channel_indices]

    def predict(
        self,
        dna_sequence: Union[str, np.ndarray, torch.Tensor],
        trans_regulator_expression: Union[np.ndarray, Sequence[float], pd.Series],
    ) -> np.ndarray:
        trans_reg_tensor = _prepare_trans_regulator_vector(trans_regulator_expression, self.config, self.device)
        seq_array = _sequence_to_array(dna_sequence)
        prediction = _predict_batch(self.model, [seq_array], trans_reg_tensor, self.device)[0]
        return self._select_channels(prediction)

    def predict_regions(
        self,
        fasta_path: Union[str, Path],
        bed_path: Union[str, Path],
        trans_regulator_expression: Union[np.ndarray, Sequence[float], pd.Series],
        batch_size: int = 2,
    ) -> np.ndarray:
        trans_reg_tensor = _prepare_trans_regulator_vector(trans_regulator_expression, self.config, self.device)
        fasta = Fasta(str(fasta_path))
        regions = _enumerate_regions(bed_path)
        sequences: List[np.ndarray] = []
        predictions: List[np.ndarray] = []

        for chrom, region_start, region_end in regions:
            for window_start, window_end in _tile_region(region_start, region_end):
                seq = fasta[chrom][window_start:window_end].seq
                sequences.append(_sequence_to_array(seq))

                if len(sequences) == batch_size:
                    predictions.append(_predict_batch(self.model, sequences, trans_reg_tensor, self.device))
                    sequences.clear()

        if sequences:
            predictions.append(_predict_batch(self.model, sequences, trans_reg_tensor, self.device))

        if not predictions:
            raise ValueError("No prediction windows produced from the supplied BED file.")
        fasta.close()
        stacked = np.concatenate(predictions, axis=0)
        return stacked[:, :, self.output_channel_indices]

    def predict_regions_with_bigwig(
        self,
        fasta_path: Union[str, Path],
        bed_path: Union[str, Path],
        chrom_sizes_path: Union[str, Path],
        trans_regulator_expression: Union[np.ndarray, Sequence[float], pd.Series],
        batch_size: int = 2,
        bigwig_dir: Union[str, Path] = "corgi_bigwig",
        bigwig_prefix: str = "corgi",
    ) -> np.ndarray:
        trans_reg_tensor = _prepare_trans_regulator_vector(trans_regulator_expression, self.config, self.device)
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
                    predictions.append(_predict_batch(self.model, sequences, trans_reg_tensor, self.device))
                    sequences.clear()

        if sequences:
            predictions.append(_predict_batch(self.model, sequences, trans_reg_tensor, self.device))

        if not predictions:
            raise ValueError("No prediction windows produced from the supplied BED file.")

        stacked = np.concatenate(predictions, axis=0)
        selected = stacked[:, :, self.output_channel_indices]
        _write_bigwig(selected, windows, chrom_sizes_path, self.output_track_names, bigwig_dir, bigwig_prefix)
        fasta.close()
        return selected


class corgiplus_pretrained:
    """Placeholder class for future Corgi+ pretrained inference support."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[dict] = None,
        output_channels: Optional[Sequence[Union[int, str]]] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.config = config
        self.output_channels = output_channels

    def predict(self, *args, **kwargs):
        raise NotImplementedError("corgiplus_pretrained is a placeholder and will be implemented later.")

    def predict_regions(self, *args, **kwargs):
        raise NotImplementedError("corgiplus_pretrained is a placeholder and will be implemented later.")

    def predict_regions_with_bigwig(self, *args, **kwargs):
        raise NotImplementedError("corgiplus_pretrained is a placeholder and will be implemented later.")


def predict_sequence(
    checkpoint_path: Union[str, Path],
    sequence: Union[str, np.ndarray, torch.Tensor],
    trans_reg_expression: Union[np.ndarray, Sequence[float], pd.Series],
    *,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[dict] = None,
    output_channels: Optional[Sequence[Union[int, str]]] = None,
) -> np.ndarray:
    predictor = corgi_pretrained(
        checkpoint_path,
        device=device,
        config=config,
        output_channels=output_channels,
    )
    return predictor.predict(sequence, trans_reg_expression)


def predict_regions(
    checkpoint_path: Union[str, Path],
    fasta_path: Union[str, Path],
    bed_path: Union[str, Path],
    trans_reg_expression: Union[np.ndarray, Sequence[float], pd.Series],
    *,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[dict] = None,
    batch_size: int = 2,
    output_channels: Optional[Sequence[Union[int, str]]] = None,
) -> np.ndarray:
    predictor = corgi_pretrained(
        checkpoint_path,
        device=device,
        config=config,
        output_channels=output_channels,
    )
    return predictor.predict_regions(fasta_path, bed_path, trans_reg_expression, batch_size=batch_size)


def predict_regions_with_bigwig(
    checkpoint_path: Union[str, Path],
    fasta_path: Union[str, Path],
    bed_path: Union[str, Path],
    chrom_sizes_path: Union[str, Path],
    trans_reg_expression: Union[np.ndarray, Sequence[float], pd.Series],
    *,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[dict] = None,
    batch_size: int = 2,
    bigwig_dir: Union[str, Path] = "corgi_bigwig",
    bigwig_prefix: str = "corgi",
    output_channels: Optional[Sequence[Union[int, str]]] = None,
) -> np.ndarray:
    predictor = corgi_pretrained(
        checkpoint_path,
        device=device,
        config=config,
        output_channels=output_channels,
    )
    return predictor.predict_regions_with_bigwig(
        fasta_path,
        bed_path,
        chrom_sizes_path,
        trans_reg_expression,
        batch_size=batch_size,
        bigwig_dir=bigwig_dir,
        bigwig_prefix=bigwig_prefix,
    )
