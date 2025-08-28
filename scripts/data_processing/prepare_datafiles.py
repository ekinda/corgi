#!/usr/bin/env python3
"""
prepare_datafiles.py

Process genomics data from h5 to npy with binning and and soft clipping

The script processes .h5 files for various experiments across multiple tissues,
bins the data into 8192 bins for every region, and
saves the resulting tensor as a float16 npy file. It also builds a one-hot
encoded DNA file and an experiment mask indicating available experiments per tissue.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import h5py
from pyfaidx import Fasta
import multiprocessing

# Global settings for binning: 524288 bp total, binsize 64 => 8192 bins.
sequence_length = 524288
bin_size = 64
n_bins = int(sequence_length / bin_size)
if sequence_length % bin_size != 0:
    raise ValueError('Sequence length is not a multiple of bin size.')

# Transformation parameters for each track.
transform_params = {
    'dnase':          {'clip': 64, 'soft_clip': 32,  'scale': 2.0,  'sum_stat': 'mean'},
    'atac':           {'clip': 64, 'soft_clip': 32,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k4me1':        {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k4me2':        {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k4me3':        {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k9ac':         {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k9me3':        {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k27ac':        {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k27me3':       {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k36me3':       {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k79me2':       {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'ctcf':           {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'cage_plus':      {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum'},
    'cage_minus':     {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum'},
    'rampage_plus':   {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum'},
    'rampage_minus':  {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum'},
    'rna_total_plus':  {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum_sqrt'},
    'rna_total_minus': {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum_sqrt'},
    'rna_polya_plus':  {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum_sqrt'},
    'rna_polya_minus': {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum_sqrt'},
    'rna_10x':         {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum_sqrt'},
    'wgbs':          {'clip': 128, 'soft_clip': 64,  'scale': 1.0,  'sum_stat': 'mean'}
}

# For 'atac', for tissue IDs 464-480, override parameters.
special_atac_params = {'clip': 64, 'soft_clip': 32, 'scale': 3.0, 'sum_stat': 'mean'}

def process_coverage(values: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply scaling, soft clip, and hard clip to coverage data.
    """
    coverage = values.astype(np.float16, copy=False)
    scale = params.get('scale', 1.0)
    soft_clip_val = params.get('soft_clip', None)
    clip_val = params.get('clip', None)

    coverage *= scale

    if soft_clip_val is not None:
        tc = float(soft_clip_val)
        coverage = np.minimum(coverage, tc + np.sqrt(np.maximum(0, coverage - tc)))
    if clip_val is not None:
        coverage = np.clip(coverage, 0, clip_val)
    return coverage

def fast_bin(raw_coverage: np.ndarray, sum_stat: str) -> np.ndarray:
    """
    Bins the raw coverage vector (length 524288) into 8192 bins (each of length 64)
    using the specified summary statistic.
    """
    reshaped = raw_coverage.reshape(n_bins, bin_size)
    if sum_stat == 'mean':
        return reshaped.mean(axis=1)
    elif sum_stat == 'sum':
        return reshaped.sum(axis=1)
    elif sum_stat == 'sum_sqrt':
        return np.sqrt(reshaped.sum(axis=1))
    else:
        raise ValueError(f"Unsupported sum_stat: {sum_stat}")
        
def load_h5_coverage(exp_file, chrom, region_start, region_end):
    """Load coverage from an .h5 file for [region_start:region_end].
       If the dataset for 'chrom' does not exist, return an array of zeros.
    """
    with h5py.File(exp_file, 'r') as hf:
        if chrom in hf:
            coverage_dataset = hf[chrom]
            return coverage_dataset[region_start:region_end]
        else:
            logging.warning(f"Dataset {chrom} not found in {exp_file}. Using zeros instead.")
            return np.zeros(region_end - region_start, dtype=np.float16)

def build_experiment_mask(encode_dir, tissue_ids, experiments):
    """
    Constructs a binary mask (tissues x experiments) indicating whether
    each experiment (genomic track) exists for a given tissue.
    """
    mask = np.zeros((len(tissue_ids), len(experiments)), dtype=np.int8)
    for i, tid in enumerate(tissue_ids):
        tissue_path = os.path.join(encode_dir, str(tid))
        if not os.path.isdir(tissue_path):
            continue
        files = [f for f in os.listdir(tissue_path) if f.endswith('.h5')]
        available = {os.path.splitext(f)[0].lower() for f in files}
        for j, exp in enumerate(experiments):
            if exp.lower() in available:
                mask[i, j] = 1
    return mask

def process_tissue(tissue_id: int, bed_df: pd.DataFrame, experiments: list, encode_dir: str, out_dir: str):
    """
    For a given tissue, loads the coverage for every available experiment,
    bins the data into 8192 bins for every region,
    and saves the resulting tensor as a float16 npy file.
    """
    out_path = os.path.join(out_dir, f"tissue_{tissue_id}.npy")
    # Skip processing if output file already exists.
    if os.path.exists(out_path):
        logging.info(f"Tissue {tissue_id}: output file {out_path} already exists, skipping.")
        return
        
    tissue_str = str(tissue_id)
    num_regions = bed_df.shape[0]

    # Identify available experiments (tracks) for this tissue.
    available_exps = []
    exp_files = []
    for exp in experiments:
        exp_file = os.path.join(encode_dir, tissue_str, f"{exp}.h5")
        if os.path.isfile(exp_file):
            available_exps.append(exp)
            exp_files.append(exp_file)
    if len(available_exps) == 0:
        logging.warning(f"Tissue {tissue_id}: no experiments found. Skipping.")
        return

    coverage_tensor = np.zeros((num_regions, n_bins, len(available_exps)), dtype=np.float16)

    for exp_index, (exp, exp_file) in enumerate(zip(available_exps, exp_files)):
        # Choose parameters; override for 'atac' if tissue_id in 464-480.
        params = transform_params.get(exp, {})
        if exp.lower() == 'atac' and 464 <= tissue_id <= 480:
            params = special_atac_params
        sum_stat = params.get('sum_stat', 'mean')

        for i, row in bed_df.iterrows():
            chrom = row['chrom']
            region_start = int(row['start'])
            region_end = int(row['end'])
            # Load and process coverage.
            raw_cov = load_h5_coverage(exp_file, chrom, region_start, region_end)

            assert len(raw_cov) == sequence_length
            
            processed = process_coverage(raw_cov, params)
            binned = fast_bin(processed, sum_stat)
            coverage_tensor[i, :, exp_index] = binned.astype(np.float16)

    out_path = os.path.join(out_dir, f"tissue_{tissue_id}.npy")
    np.save(out_path, coverage_tensor, allow_pickle=False)
    logging.info(f"Tissue {tissue_id}: saved coverage with shape {coverage_tensor.shape} to {out_path}")

def one_hot_encode(seq: str) -> np.ndarray:
    """
    One-hot encodes a DNA sequence.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    onehot = np.zeros((len(seq), 4), dtype=np.int8)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            onehot[i, mapping[base]] = 1
    return onehot

def build_onehot_file(ref_fasta: str, bed_df: pd.DataFrame, output_path: str):
    """
    Builds a one-hot encoded DNA file from a fasta file and bed regions.
    """
    ref = Fasta(ref_fasta)
    num_regions = bed_df.shape[0]
    seq_length = sequence_length
    onehot_array = np.zeros((num_regions, seq_length, 4), dtype=np.int8)

    for i, row in bed_df.iterrows():
        chrom = row['chrom']
        region_start = int(row['start'])
        region_end = int(row['end'])

        assert region_end - region_start == sequence_length
        
        seq = ref[chrom][region_start:region_end].seq
        onehot_array[i] = one_hot_encode(seq)
        if (i + 1) % 1000 == 0:
            logging.info(f"One-hot encoded {i+1}/{num_regions} regions")
            
    ref.close()
    np.save(output_path, onehot_array, allow_pickle=False)
    logging.info(f"Saved one-hot encoded DNA to {output_path} with shape {onehot_array.shape}")

def run_process_tissue(args):
    tissue_id, bed_df, experiments, encode_dir, out_dir = args
    try:
        process_tissue(tissue_id, bed_df, experiments, encode_dir, out_dir)
    except Exception as e:
        logging.error(f"Error processing tissue {tissue_id}: {e}", exc_info=True)
    else:
        logging.info(f"Finished processing tissue {tissue_id}")

def main():
    parser = argparse.ArgumentParser(description="Process genomics data from h5 to npy with binning and one-hot encoding.")
    parser.add_argument("--encode_dir", required=True, help="Directory containing tissue subdirectories with H5 files.")
    parser.add_argument("--bed_file", required=True, help="BED file with regions (columns: chrom, start, end, fold).")
    parser.add_argument("--ref_fasta", required=True, help="Reference genome FASTA for one-hot encoding.")
    parser.add_argument("--output_dir", required=True, help="Output directory for npy files, experiment mask, and one-hot file.")
    parser.add_argument("--start", required=True, type=int, help="Start tissue ID.")
    parser.add_argument("--end", required=True, type=int, help="End tissue ID.")
    parser.add_argument("--threads", type=int, default=1, help="Number of parallel processes (default 1).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    # Load experiments list.
    experiments = list(transform_params.keys())
    print('Experiment tracks in order:', experiments)

    # Load BED file.
    bed_df = pd.read_csv(args.bed_file, sep="\t", header=None, names=["chrom", "start", "end", "fold"])
    logging.info(f"Loaded {bed_df.shape[0]} regions from BED file.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build one-hot encoded DNA file.
    onehot_path = os.path.join(args.output_dir, "dna_onehot.npy")
    if not os.path.exists(onehot_path):
        logging.info("Building one-hot encoded DNA file...")
        build_onehot_file(args.ref_fasta, bed_df, onehot_path)
    else:
        logging.info(f"{onehot_path} exists; skipping one-hot build.")

    # Build experiment mask.
    tissue_ids = list(range(args.start, args.end + 1))
    logging.info("Building experiment mask...")
    mask = build_experiment_mask(args.encode_dir, tissue_ids, experiments)
    mask_path = os.path.join(args.output_dir, "experiment_mask.npy")
    np.save(mask_path, mask, allow_pickle=False)
    logging.info(f"Saved experiment mask to {mask_path}")

    # Process tissues in parallel.
    tasks = [(tid, bed_df, experiments, args.encode_dir, args.output_dir) for tid in tissue_ids]
    logging.info(f"Processing tissues {args.start} to {args.end} using {args.threads} threads.")
    if args.threads > 1:
        with multiprocessing.Pool(args.threads) as pool:
            pool.map(run_process_tissue, tasks)
    else:
        for t in tasks:
            run_process_tissue(t)
    logging.info("All tissues processed successfully.")

if __name__ == "__main__":
    main()
