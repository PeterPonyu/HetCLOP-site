"""The 50 dataset names (ORIGINAL 15 + NEW 35), copied verbatim from
experiments/real_phase0_extended_2026-07-08/combine_and_score.py so this
experiment reuses the exact same corpus. Also the cached-FM-embedding path
resolver (reused for Arm A, which decodes cached full-gene FM embeddings)."""
from __future__ import annotations

from pathlib import Path

HETCLOP_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_RESULTS_DIR = HETCLOP_ROOT / "experiments" / "real_phase0_2026-07-05" / "results"
EXTENDED_RESULTS_DIR = HETCLOP_ROOT / "experiments" / "real_phase0_extended_2026-07-08" / "results"
UCE_LEGACY_CACHE_DIR = HETCLOP_ROOT / "data" / "embeddings" / "fm_emb"

ORIGINAL_DATASETS = [
    "GSE115571_LPSMmDev", "GSE145929_ProgastinMmDev", "GSE145929_UrineMmDev",
    "GSE165844_LSKMmBatch", "GSE167597_spineMm", "GSE189070_astrocytesSCIMmDev",
    "GSE226131_HSCMmAged", "GSE275119_TeethMmDev",
    "GSE132509_acutelymluekPBMCHmCancer", "GSE123902_LungAdreHmCancer",
    "GSE165784_RetinaHmDev", "GSE192857_hESCHmTimes", "GSE213740_ADHm",
    "GSE222002_TcellsHmCancer", "GSE98638_TcellLiverHmCancer",
]
NEW_DATASETS = [
    "GSE117988_MCCPBMCCancer", "GSE117988_MCCTumorCancer", "GSE120505_bloodAged",
    "GSE123813_bccHmCancer", "GSE123813_sccHmCancer", "GSE124310_MMHmCancer",
    "GSE130148_LungHmDev", "GSE138709_LiverCancer", "GSE142653pitHmDev",
    "GSE143423_lbm_CancerBrainHm", "GSE143423_tnbc_CancerBrainHm",
    "GSE148218_bmALLHmCancer", "GSE149655_CAHmCancer", "GSE155109_bcECHmCancer",
    "GSE155109_bcStromaHmCancer", "GSE163558_stomachHmCancer", "GSE168181_BreastHmCancer",
    "GSE183904_GastricHmCancer", "GSE189357_lungAdreHmCancer",
    "GSE222369_NKsLymphomaHmCancer", "GSE225600_breast_CancerHm",
    "GSE225857_liverColonMetasisHmCancer", "GSE228499_breastHmCancer",
    "GSE235787_bcellsALLHmCancer", "GSE253355_bmNicheHm", "GSE262288_breastMetasisHmCancer",
    "GSE283205_hepatoblastomaCancer", "bm_GSE120446", "dentate", "endo",
    "hESC_GSE144024", "hemato", "ifnHSPC_GSE226824", "lung", "setty",
]
ALL_DATASETS = ORIGINAL_DATASETS + NEW_DATASETS


def fm_cache_path(fm_name: str, dataset_name: str) -> Path:
    """Resolve a cached full-gene FM embedding npz (Arm A reuse). Mirrors
    experiments/real_phase0_extended_2026-07-08/combine_and_score.py."""
    if dataset_name in ORIGINAL_DATASETS:
        if fm_name == "uce":
            return UCE_LEGACY_CACHE_DIR / f"uce_uce_{dataset_name}.npz"
        return ORIGINAL_RESULTS_DIR / f"{fm_name}_{dataset_name}.npz"
    return EXTENDED_RESULTS_DIR / f"{fm_name}_{dataset_name}.npz"
