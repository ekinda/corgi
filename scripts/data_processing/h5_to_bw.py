#!/usr/bin/env python3
"""
h5_to_bw.py

Convert root-level chromosome datasets in .h5 files back into BigWig format.
Traverses an input directory recursively, mirrors the directory structure in an
output directory, and writes a .bw file for every .h5 file found.
"""

import argparse
import logging
import os
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import pyBigWig

# Default chunk size when streaming per-base values into BigWig.
DEFAULT_CHUNK_SIZE = 100_000


def adjust_chrom_name(raw: str, add_chr: bool, strip_chr: bool) -> str:
    """Normalize chromosome names according to flags."""
    if add_chr and raw.startswith("chr"):
        return raw
    if strip_chr and raw.startswith("chr"):
        return raw[3:]
    if add_chr and not raw.startswith("chr"):
        return f"chr{raw}"
    return raw


def discover_h5_files(input_dir: str) -> Iterable[str]:
    """Yield absolute paths to .h5 files under input_dir (recursive)."""
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.endswith(".h5"):
                yield os.path.join(root, fname)


def build_header(h5_handle: h5py.File, add_chr: bool, strip_chr: bool) -> Tuple[List[Tuple[str, int]], Dict[str, str]]:
    """Collect BigWig header entries and map raw dataset names to output chrom names."""
    header: List[Tuple[str, int]] = []
    chrom_map: Dict[str, str] = {}
    for key in h5_handle.keys():
        chrom_out = adjust_chrom_name(key, add_chr, strip_chr)
        chrom_len = len(h5_handle[key])
        header.append((chrom_out, chrom_len))
        chrom_map[key] = chrom_out
    return header, chrom_map


def write_bigwig(h5_path: str, bw_path: str, add_chr: bool, strip_chr: bool, chunk_size: int) -> None:
    """Convert a single h5 coverage file to BigWig."""
    with h5py.File(h5_path, "r") as hf:
        header, chrom_map = build_header(hf, add_chr, strip_chr)
        with pyBigWig.open(bw_path, "w") as bw:
            bw.addHeader(header)
            for raw_chrom, ds in hf.items():
                chrom = chrom_map[raw_chrom]
                values = np.asarray(ds[:], dtype=np.float64)
                if np.isnan(values).any():
                    values = np.nan_to_num(values)
                total = values.shape[0]
                for start in range(0, total, chunk_size):
                    chunk = values[start:start + chunk_size]
                    if chunk.size == 0:
                        continue
                    starts = np.arange(start, start + chunk.size, dtype=np.int64)
                    bw.addEntries(chrom, starts.tolist(), chunk.tolist())
        logging.info("Wrote %s", bw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert coverage .h5 files to BigWig, recursively.")
    parser.add_argument("input_dir", help="Directory containing .h5 files (searched recursively).")
    parser.add_argument("output_dir", help="Directory to mirror structure and write .bw files.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Number of bases to stream per addEntries call (default: %(default)s).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .bw files.")
    chrom_group = parser.add_mutually_exclusive_group()
    chrom_group.add_argument("--add-chr-prefix", action="store_true",
                             help="Prefix chromosome names with 'chr' when missing.")
    chrom_group.add_argument("--strip-chr-prefix", action="store_true",
                             help="Remove leading 'chr' from chromosome names.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: %(default)s).")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(asctime)s] %(levelname)s: %(message)s")

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    h5_files = list(discover_h5_files(args.input_dir))
    if not h5_files:
        logging.warning("No .h5 files found under %s", args.input_dir)
        return

    for h5_path in h5_files:
        rel = os.path.relpath(h5_path, args.input_dir)
        rel_dir = os.path.dirname(rel)
        out_dir = os.path.join(args.output_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.splitext(os.path.basename(h5_path))[0] + ".bw"
        bw_path = os.path.join(out_dir, out_name)
        if os.path.exists(bw_path) and not args.overwrite:
            logging.info("Skipping existing %s", bw_path)
            continue
        try:
            write_bigwig(
                h5_path=h5_path,
                bw_path=bw_path,
                add_chr=args.add_chr_prefix,
                strip_chr=args.strip_chr_prefix,
                chunk_size=max(1, args.chunk_size),
            )
        except Exception:
            logging.exception("Failed to convert %s", h5_path)


if __name__ == "__main__":
    main()
