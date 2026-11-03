"""Human-mouse gene-symbol ortholog resolution.

The real ortholog table (mouse symbol -> human symbol) should be sourced from
MGI's HOM_MouseHumanSequence.rpt (https://www.informatics.jax.org/downloads/reports/)
or a locally available equivalent, converted to the two-column TSV
(`mouse_symbol`, `human_symbol`) expected by `load_ortholog_table`. This module
only implements the lookup logic; populating `data/orthologs/human_mouse_orthologs.tsv`
with the real MGI table is a data-acquisition step to run once before Phase 0
execution, not part of this infrastructure task.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_symbol(sym: str) -> str:
    return sym.strip().upper()


def load_ortholog_table(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t")
    return {
        normalize_symbol(mouse): normalize_symbol(human)
        for mouse, human in zip(df["mouse_symbol"], df["human_symbol"])
    }


def map_mouse_symbols_to_human(
    mouse_symbols: list[str], ortholog_table: dict[str, str]
) -> list[str | None]:
    return [ortholog_table.get(normalize_symbol(s)) for s in mouse_symbols]
