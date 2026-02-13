# Corgi

Corgi is a deep neural network that predicts genomic sequencing tracks such as DNase-seq, ATAC-seq, histone ChIP-seq, CTCF binding and DNA methylation.

See the following preprint for details:

https://www.biorxiv.org/content/10.1101/2025.06.25.661447v1

Model weights can be downloaded from: https://zenodo.org/records/17368602

## Tutorial

### 1) System requirements

- Linux (recommended)
- Python >= 3.9
- NVIDIA GPU with **Ampere or newer** architecture (A100, RTX 30xx, RTX 40xx, H100, etc.)
- CUDA-enabled PyTorch installation compatible with your driver

> Why Ampere+? Corgi uses FlashAttention v2 rotary positional encodings and bfloat16 autocast for fast inference. Older GPUs are not supported for the intended inference path.

### 2) Package requirements

Core Python dependencies:

- `torch>=2.1`
- `flash-attn>=2.0.0`
- `numpy>=1.24`
- `pandas>=1.5`
- `pyfaidx>=0.7`
- `pybigwig>=0.3.22`

### 3) Installation

Package-install placeholder (name may change later):

```bash
pip install corgi
```

Install from source:

```bash
git clone https://github.com/ekinda/corgi.git
cd corgi
pip install .
```

If you prefer editable install for development:

```bash
pip install -e .
```

### 4) Download checkpoints

- Corgi checkpoint: https://zenodo.org/records/17368602

### 5) Quick single-sequence inference

```python
import numpy as np
from corgi.predict import corgi_pretrained

# 524,288bp DNA sequence (string A/C/G/T/N) or one-hot array with shape (524288, 4)
dna_sequence = "A" * 524_288

# Trans-regulator expression in expected Corgi order (length 2891),
# or a pandas Series indexed by HGNC/ENSG symbols.
trans_reg = np.zeros(2891, dtype=np.float32)

model = corgi_pretrained(
	checkpoint_path="/path/to/corgi_checkpoint.pt",
	device="cuda",
)

pred = model.predict(dna_sequence, trans_reg)
print(pred.shape)  # (6144, 22) by default
```

### 6) Region-based prediction from FASTA + BED

```python
pred_regions = model.predict_regions(
	fasta_path="/path/to/genome.fa",
	bed_path="/path/to/regions.bed",
	trans_regulator_expression=trans_reg,
	batch_size=2,
)
print(pred_regions.shape)  # (num_windows, 6144, channels)
```

### 7) BigWig export

```python
pred_regions = model.predict_regions_with_bigwig(
	fasta_path="/path/to/genome.fa",
	bed_path="/path/to/regions.bed",
	trans_regulator_expression=trans_reg,
	batch_size=2,
	bigwig_dir="corgi_bigwig",
	bigwig_prefix="sample1",
)
```

This writes one BigWig file per selected output channel.

### 8) Selecting output channels

`corgi_pretrained(..., output_channels=...)` supports:

- `None` (default): all tracks from `corgi/constants.py`
- list of indices, e.g. `[0, 11, 20]`
- list of names, e.g. `["dnase", "ctcf", "rna_10x"]`

Returned tensors and BigWig outputs will follow only those selected channels.
