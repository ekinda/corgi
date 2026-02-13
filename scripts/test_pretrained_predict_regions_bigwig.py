import argparse
import tempfile
from pathlib import Path

import numpy as np

from corgi.config import config_corgi
from corgi.predict import corgi_pretrained


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Corgi pretrained region+BigWig test")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Corgi checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device, e.g. cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--fasta",
        type=str,
        default="/project/deeprna_data/corgi-reproduction/data/hg38.ml.fa",
        help="Path to hg38 FASTA used for region extraction",
    )
    parser.add_argument(
        "--chrom-sizes",
        type=str,
        default="/project/deeprna/corgi/data/hg38.chrom.sizes",
        help="Path to chrom.sizes file used to build BigWig headers",
    )
    parser.add_argument(
        "--bed",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "test_predict_regions.bed"),
        help="BED file with test regions",
    )
    args = parser.parse_args()

    bed_path = Path(args.bed)
    fasta_path = Path(args.fasta)
    chrom_sizes_path = Path(args.chrom_sizes)
    trans_reg = np.zeros(config_corgi["input_trans_regulators"], dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="corgi_bigwig_test_") as tmp_dir:
        bigwig_dir = Path(tmp_dir) / "bigwig_out"

        model = corgi_pretrained(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )

        pred = model.predict_regions_with_bigwig(
            fasta_path=fasta_path,
            bed_path=bed_path,
            chrom_sizes_path=chrom_sizes_path,
            trans_regulator_expression=trans_reg,
            batch_size=args.batch_size,
            bigwig_dir=bigwig_dir,
            bigwig_prefix="test",
        )

        bigwig_files = sorted(bigwig_dir.glob("*.bw"))
        print("prediction shape:", pred.shape)
        print("bigwig files written:", len(bigwig_files))
        for path in bigwig_files[:5]:
            print(" -", path.name)


if __name__ == "__main__":
    main()
