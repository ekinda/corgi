import argparse
import numpy as np
import torch

from corgi.config import config_corgi
from corgi.predict import corgi_pretrained

def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Corgi pretrained inference test")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Corgi checkpoint (.pt)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Using GPU for inference.")
    else:
        raise RuntimeError("CUDA is not available. Please run on a machine with a compatible NVIDIA GPU.")

    dna_sequence = "A" * 524_288
    trans_reg = np.zeros(config_corgi["input_trans_regulators"], dtype=np.float32)

    model = corgi_pretrained(
        checkpoint_path=args.checkpoint,
        device='cuda',
    )

    pred = model.predict(dna_sequence, trans_reg)
    print("prediction shape:", pred.shape)

if __name__ == "__main__":
    main()
