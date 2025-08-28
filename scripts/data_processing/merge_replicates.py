#!/usr/bin/env python3
"""
merge_replicates.py

This script processes sample subdirectories from an input directory
and writes merged replicate .h5 files into corresponding subdirectories in an output directory.

Input directory should contain sample directories, each with .h5 files for various experiments.
The h5 files in the input directory should be of form: {experiment}*.h5
Example: dnase_s1_r1.h5, dnase_s1_r2.h5, dnase_s2_r1.h5, etc.


For each sample directory, the script:
  - Scans for .h5 files matching a set of experiments.
  - For experiments with multiple replicates, calls an external merge script (h5_merge.py)
    to merge the files (default is mean signal).
  - If only one file is found, the file is copied to the output directory with no mofidications.
  - Special rules:
      - For "cage": All files with "cage" in their name are grouped by strand ("plus" and "minus")
        and merged as cage_plus.h5 and cage_minus.h5.
      - For "atac": All files containing "atac" in the name (covering both atac_s1_r1 and atac_10x)
        are merged as atac.h5.
        
Usage:
    python merge_replicates_parallel.py input_dir output_dir [--ncores N]

Example:
    python merge_replicates_parallel.py ./raw_data ./merged_data --ncores 8
"""

import os
import glob
import argparse
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# Path to the external merge script
MERGE_SCRIPT = "./h5_merge.py"

# Predefined list of experiments
EXPERIMENTS = [
    "dnase",
    "atac",
    "h3k4me1",
    "h3k4me2",
    "h3k4me3",
    "h3k9ac",
    "h3k9me3",
    "h3k27ac",
    "h3k27me3"
    "h3k36me3",
    "h3k79me2",
    "ctcf",
    "cage_plus",
    "cage_minus",
    "rampage_plus",
    "rampage_minus",
    "rna_total_plus",
    "rna_total_minus",
    "rna_polya_plus",
    "rna_polya_minus",
    "rna_10x",
    "wgbs"
]

def run_merge(file_list, output_file):
    """
    Call the external merge script with the specified list of .h5 files.
    The merge script is called with the following options:
      - -w : overwrite existing file
      - -s mean : merge strategy using the mean
      - -z <output_file> : output file name
    """
    cmd = ["python", MERGE_SCRIPT, "-w", "-s", "mean", "-z", output_file] + file_list
    print("Running merge command: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"Merged {len(file_list)} files into {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error merging into {output_file}: {e}")

def process_folder(input_folder, output_folder):
    """
    Process a single tissue folder by:
      - Creating the corresponding output folder.
      - Deciding which .h5 files to merge based on experiment names.
      - Handling special cases for "cage" and "atac".
    """
    print(f"Processing tissue folder: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    # ----- Special Handling for Cage -----
    cage_files = glob.glob(os.path.join(input_folder, "*cage*.h5"))
    cage_plus_files = [f for f in cage_files if "plus" in os.path.basename(f).lower()]
    cage_minus_files = [f for f in cage_files if "minus" in os.path.basename(f).lower()]
    
    if cage_plus_files:
        out_name = os.path.join(output_folder, "cage_plus.h5")
        if len(cage_plus_files) > 1:
            run_merge(cage_plus_files, out_name)
        else:
            src = cage_plus_files[0]
            if os.path.basename(src) != "cage_plus.h5":
                shutil.copy(src, out_name)
                print(f"Copied {src} to {out_name}")
            else:
                shutil.copy(src, out_name)
                print(f"Copied {src} to {out_name}")
    
    if cage_minus_files:
        out_name = os.path.join(output_folder, "cage_minus.h5")
        if len(cage_minus_files) > 1:
            run_merge(cage_minus_files, out_name)
        else:
            src = cage_minus_files[0]
            if os.path.basename(src) != "cage_minus.h5":
                shutil.copy(src, out_name)
                print(f"Copied {src} to {out_name}")
            else:
                shutil.copy(src, out_name)
                print(f"Copied {src} to {out_name}")
    
    # ----- Special Handling for Atac -----
    atac_files = glob.glob(os.path.join(input_folder, "*atac*.h5"))
    if atac_files:
        out_name = os.path.join(output_folder, "atac.h5")
        if len(atac_files) > 1:
            run_merge(atac_files, out_name)
        else:
            src = atac_files[0]
            if os.path.basename(src) != "atac.h5":
                shutil.copy(src, out_name)
                print(f"Copied {src} to {out_name}")
            else:
                shutil.copy(src, out_name)
                print(f"Copied {src} to {out_name}")
    
    # ----- Standard Handling for Other Experiments -----
    # Skip the special ones that are already handled: "cage_plus", "cage_minus", "atac"
    for exp in EXPERIMENTS:
        if exp in ["cage_plus", "cage_minus", "atac"]:
            continue
        pattern = os.path.join(input_folder, f"{exp}*.h5")
        exp_files = glob.glob(pattern)
        if not exp_files:
            continue
        out_name = os.path.join(output_folder, f"{exp}.h5")
        if len(exp_files) > 1:
            run_merge(exp_files, out_name)
        else:
            src = exp_files[0]
            if os.path.basename(src) != f"{exp}.h5":
                shutil.copy(src, out_name)
                print(f"Copied {src} to {out_name}")
            else:
                shutil.copy(src, out_name)
                print(f"Copied {src} to {out_name}")
    
    print(f"Completed processing folder: {input_folder} -> {output_folder}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge replicate .h5 files from tissue folders using an external merge script, "
                    "writing outputs into a separate directory structure."
    )
    parser.add_argument("input_dir", help="Directory containing tissue subfolders with .h5 files")
    parser.add_argument("output_dir", help="Directory where merged tissue folders will be created")
    parser.add_argument("--ncores", type=int, default=8, help="Number of parallel processes (default: 8)")
    args = parser.parse_args()
    
    tissue_folders = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir)
                      if os.path.isdir(os.path.join(args.input_dir, d))]
    
    if not tissue_folders:
        print(f"No subdirectories found in {args.input_dir}.")
        return
    
    with ProcessPoolExecutor(max_workers=args.ncores) as executor:
        futures = {}
        for folder in tissue_folders:
            out_folder = os.path.join(args.output_dir, os.path.basename(folder))
            futures[executor.submit(process_folder, folder, out_folder)] = folder
        
        for future in as_completed(futures):
            folder = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Error processing {folder}: {exc}")

if __name__ == "__main__":
    main()
