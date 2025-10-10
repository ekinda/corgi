"""Helpers for ordering trans regulator expression vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import config_corgi


def load_trans_regulator_lists(config: Optional[dict] = None) -> Tuple[Sequence[str], Sequence[str]]:
    cfg = config or config_corgi
    symbols_path = Path(cfg["trans_regulator_symbols_path"])
    ensembl_path = Path(cfg["trans_regulator_ensembl_path"])
    with symbols_path.open("r", encoding="utf-8") as handle:
        symbols = [line.strip() for line in handle if line.strip()]
    with ensembl_path.open("r", encoding="utf-8") as handle:
        ensembl = [line.strip() for line in handle if line.strip()]
    return symbols, ensembl


def encode_trans_reg_expression(
    expression: Union[np.ndarray, Sequence[float], pd.Series],
    config: Optional[dict] = None,
) -> np.ndarray:
    cfg = config or config_corgi
    expected = cfg["input_trans_regulators"]
    if isinstance(expression, pd.Series):
        symbols, ensembl = load_trans_regulator_lists(cfg)
        index_values = expression.index.astype(str)
        if len(index_values) and index_values[0].upper().startswith("ENSG"):
            order = ensembl
        else:
            order = symbols
        values = expression.reindex(order).fillna(0.0).to_numpy(dtype=np.float32)
    else:
        values = np.asarray(expression, dtype=np.float32)
    if values.shape[0] != expected:
        raise ValueError(f"Expression vector must have length {expected} (received {values.shape[0]}).")
    return values
