from src.orthologs.mapping import normalize_symbol, map_mouse_symbols_to_human


def test_normalize_symbol_uppercases_and_strips():
    assert normalize_symbol(" Actb ") == "ACTB"


def test_map_mouse_symbols_to_human_known_orthologs():
    # Real biological convention: mouse gene symbols are TitleCase (Actb, Tp53),
    # human orthologs of the same gene are ALL CAPS (ACTB, TP53).
    table = {"ACTB": "ACTB", "TP53": "TP53", "GAPDH": "GAPDH"}
    result = map_mouse_symbols_to_human(["Actb", "Tp53", "NotAGene"], table)
    assert result == ["ACTB", "TP53", None]


def test_map_mouse_symbols_to_human_empty_list():
    assert map_mouse_symbols_to_human([], {"ACTB": "ACTB"}) == []
