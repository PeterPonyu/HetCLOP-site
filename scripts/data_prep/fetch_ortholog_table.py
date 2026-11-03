"""Downloads MGI's real human-mouse ortholog table and writes it to
data/orthologs/human_mouse_orthologs.tsv, the format
src.orthologs.mapping.load_ortholog_table expects. Idempotent: skips
re-downloading if the output file already exists, unless --force is passed.
URL confirmed reachable from this environment during design (HTTP 200,
~15MB) -- see the design notes
design.md section 4."""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.orthologs.mgi_parser import parse_hom_mouse_human
from src.utils.paths import PROJECT_ROOT

MGI_URL = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "orthologs" / "human_mouse_orthologs.tsv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        print(f"{OUTPUT_PATH} already exists; skipping download (use --force to re-fetch)")
        return

    print(f"Downloading {MGI_URL} ...")
    with urllib.request.urlopen(MGI_URL, timeout=60) as response:
        raw_text = response.read().decode("utf-8")

    df = parse_hom_mouse_human(raw_text)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"Wrote {len(df)} mouse->human ortholog pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
