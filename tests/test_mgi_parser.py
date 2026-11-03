# tests/test_mgi_parser.py
import pandas as pd
import pytest

from src.orthologs.mgi_parser import parse_hom_mouse_human


FIXTURE_TEXT = (
    "DB Class Key\tCommon Organism Name\tNCBI Taxon ID\tSymbol\tEntrezGene ID\t"
    "Mouse MGI ID\tHGNC ID\tOMIM Gene ID\tGenetic Location\t"
    "Genome Coordinates (mouse: GRCm39 human: GRCh38)\tNucleotide RefSeq IDs\t"
    "Protein RefSeq IDs\tSWISS_PROT IDs\n"
    "51743063\tmouse, laboratory\t10090\tAldh1l1\t107747\tMGI:1340024\t\t\t"
    "Chr6 40.16 cM\tChr6:90527751-90576153(+)\tXM_030255052,NM_027406,NM_001356412\t"
    "XP_030110912,NP_081682,NP_001343341\tQ8R0Y6\n"
    "51743063\thuman\t9606\tALDH1L1\t10840\t\tHGNC:3978\tOMIM:600249\tChr3 q21.3\t"
    "Chr3:126103561-126197994(-)\tNM_012190,NM_001270364,NM_001270365,NM_144776\t"
    "XP_024309093\tO75891\n"
    "51743064\tmouse, laboratory\t10090\tSry\t21674\tMGI:98660\t\t\tChrY syntenic\t"
    "ChrY:2662471-2663658(-)\tNM_011564\tNP_035694\tQ05738\n"
    "51743064\thuman\t9606\tSRY\t6736\t\tHGNC:11311\tOMIM:480000\tChrY p11.2\t"
    "ChrY:2786855-2787682(-)\tNM_003140\tNP_003131\tQ05066\n"
    "99999999\tmouse, laboratory\t10090\tGeneX\t1\tMGI:1\t\t\tChr1\tChr1:1-2(+)\t"
    "NM_1\tNP_1\tP1\n"
    "99999999\tmouse, laboratory\t10090\tGeneY\t2\tMGI:2\t\t\tChr1\tChr1:3-4(+)\t"
    "NM_2\tNP_2\tP2\n"
    "99999999\thuman\t9606\tGENEZ\t3\t\tHGNC:1\tOMIM:1\tChr1\tChr1:5-6(-)\t"
    "NM_3\tNP_3\tP3"
)


def test_parse_hom_mouse_human_extracts_1to1_pairs():
    df = parse_hom_mouse_human(FIXTURE_TEXT)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["mouse_symbol", "human_symbol"]
    result = dict(zip(df["mouse_symbol"], df["human_symbol"]))
    assert result == {"Aldh1l1": "ALDH1L1", "Sry": "SRY"}


def test_parse_hom_mouse_human_skips_many_to_many_groups():
    df = parse_hom_mouse_human(FIXTURE_TEXT)
    assert "GeneX" not in list(df["mouse_symbol"])
    assert "GeneY" not in list(df["mouse_symbol"])
    assert "GENEZ" not in list(df["human_symbol"])


def test_parse_hom_mouse_human_returns_exactly_two_rows_for_fixture():
    df = parse_hom_mouse_human(FIXTURE_TEXT)
    assert len(df) == 2
