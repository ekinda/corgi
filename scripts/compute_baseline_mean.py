import argparse
import logging
from pathlib import Path

import numpy as np

from corgi.utils import load_experiment_mask


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')


def _load_ids(path: Path):
    with open(path) as f:
        return [int(x) for x in f.read().strip().split() if x.strip()]


def compute_baseline(input_dir: Path, mask_path: Path, tissue_ids_path: Path, output_path: Path, output_channels: int, dtype: str = "float32"):
    tissue_ids = _load_ids(tissue_ids_path)
    if not tissue_ids:
        raise ValueError("No tissue ids provided.")

    mask = load_experiment_mask(str(mask_path))
    first_file = input_dir / f"tissue_{tissue_ids[0]}.npy"
    if not first_file.exists():
        raise FileNotFoundError(f"Cannot find {first_file}")

    sample = np.load(first_file, mmap_mode='r')
    num_seqs, seq_len, avail_channels = sample.shape
    logging.info(f"Found {num_seqs} sequences, length {seq_len}, channels {avail_channels} in {first_file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use open_memmap to create valid .npy files with headers (np.memmap alone writes raw binary).
    baseline = np.lib.format.open_memmap(output_path, mode='w+', dtype=dtype, shape=(num_seqs, seq_len, output_channels))
    counts_path = output_path.with_suffix(output_path.suffix + '.counts.npy')
    counts = np.lib.format.open_memmap(counts_path, mode='w+', dtype='uint16', shape=(num_seqs, output_channels))
    baseline[:] = 0
    counts[:] = 0

    for tid in tissue_ids:
        tissue_file = input_dir / f"tissue_{tid}.npy"
        if not tissue_file.exists():
            logging.warning(f"Skipping missing file {tissue_file}")
            continue

        data = np.load(tissue_file, mmap_mode='r')
        if data.shape[0] != num_seqs or data.shape[1] != seq_len:
            raise ValueError(f"Shape mismatch for {tissue_file}: expected ({num_seqs}, {seq_len}, *), got {data.shape}")

        tissue_mask = mask.get(tid)
        if tissue_mask is None:
            raise KeyError(f"No mask entry for tissue {tid}")

        available = np.where(tissue_mask == 1)[0]
        if data.shape[2] != len(available):
            logging.warning(f"Channel count mismatch for tissue {tid}: data channels={data.shape[2]}, mask available={len(available)}")

        baseline[:, :, available] += data
        counts[:, available] += 1
        logging.info(f"Accumulated tissue {tid}")

    denom = np.maximum(counts, 1)[:, None, :]
    baseline /= denom

    baseline.flush(); counts.flush()
    # Explicitly close memmaps
    del baseline, counts
    logging.info(f"Baseline saved to {output_path} and counts to {counts_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute per-sequence baseline mean across tissues.")
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory containing tissue_{id}.npy files')
    parser.add_argument('--mask-path', type=Path, required=True, help='Experiment mask npy path')
    parser.add_argument('--tissue-ids-path', type=Path, required=True, help='Text file with tissue ids to average (space/newline separated)')
    parser.add_argument('--output-path', type=Path, required=True, help='Where to write the baseline .npy')
    parser.add_argument('--output-channels', type=int, default=22, help='Total output channels (default 22)')
    parser.add_argument('--dtype', type=str, default='float32', help='Output dtype (default float32)')

    args = parser.parse_args()
    compute_baseline(args.input_dir, args.mask_path, args.tissue_ids_path, args.output_path, args.output_channels, args.dtype)


if __name__ == '__main__':
    main()
