"""Parses MGI's HOM_MouseHumanSequence.rpt (real, public, tab-delimited
ortholog report -- format confirmed by downloading and inspecting the actual
file during design, see the design notes
readiness-design.md section 4) into the two-column (mouse_symbol,
human_symbol) table src.orthologs.mapping.load_ortholog_table expects."""
from __future__ import annotations

import pandas as pd


def parse_hom_mouse_human(raw_text: str) -> pd.DataFrame:
    lines = [line for line in raw_text.split("\n") if line]
    header = lines[0].split("\t")
    key_idx = header.index("DB Class Key")
    organism_idx = header.index("Common Organism Name")
    symbol_idx = header.index("Symbol")

    groups: dict[str, list[tuple[str, str]]] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        key = fields[key_idx]
        organism = fields[organism_idx]
        symbol = fields[symbol_idx]
        groups.setdefault(key, []).append((organism, symbol))

    mouse_symbols_out: list[str] = []
    human_symbols_out: list[str] = []
    for entries in groups.values():
        mouse_symbols = [s for org, s in entries if org.startswith("mouse")]
        human_symbols = [s for org, s in entries if org == "human"]
        if len(mouse_symbols) == 1 and len(human_symbols) == 1:
            mouse_symbols_out.append(mouse_symbols[0])
            human_symbols_out.append(human_symbols[0])

    return pd.DataFrame({"mouse_symbol": mouse_symbols_out, "human_symbol": human_symbols_out})
