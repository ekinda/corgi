"""corgi – Genomic track prediction package."""

from .config import config_corgi
from .finetune import FinetuneSettings, finetune_corgi
from .model import Corgi, CorgiPlus
from .predict import corgi_pretrained, corgiplus_pretrained, predict_regions, predict_regions_with_bigwig, predict_sequence

__all__ = [
    "Corgi",
    "CorgiPlus",
    "corgi_pretrained",
    "corgiplus_pretrained",
    "config_corgi",
    "FinetuneSettings",
    "finetune_corgi",
    "predict_sequence",
    "predict_regions",
    "predict_regions_with_bigwig",
]
