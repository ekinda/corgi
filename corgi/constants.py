"""Shared constants used across the corgi package."""

from __future__ import annotations

WINDOW_SIZE: int = 524_288
STRIDE: int = 393_216
BIN_SIZE: int = 64
CENTRAL_BINS: int = 6_144
CENTRAL_LENGTH: int = CENTRAL_BINS * BIN_SIZE
FLANK: int = (WINDOW_SIZE - CENTRAL_LENGTH) // 2

DEFAULT_TRACK_NAMES = (
    "dnase",
    "atac",
    "h3k4me1",
    "h3k4me2",
    "h3k4me3",
    "h3k9ac",
    "h3k9me3",
    "h3k27ac",
    "h3k27me3",
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
    "wgbs",
)
