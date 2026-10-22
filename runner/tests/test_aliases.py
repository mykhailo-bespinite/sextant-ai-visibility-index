import pytest

from aeo_index.aliases import KNOWN_TOOLS, AliasTable, normalise

PUBLISHED_ORDER = [
    "Profound",
    "Peec AI",
    "Otterly",
    "Scrunch",
    "RankSpot",
    "OpenSEO",
    "Semrush AI Toolkit",
    "Ahrefs Brand Radar",
    "Similarweb",
    "Conductor",
    "BrightEdge",
    "Sextant AEO",
]


def test_table_is_the_published_one_in_order():
    assert list(KNOWN_TOOLS) == PUBLISHED_ORDER
    assert KNOWN_TOOLS["Peec AI"] == ("Peec AI", "Peec")
    assert KNOWN_TOOLS["Otterly"] == ("Otterly", "Otterly.AI", "Otterly AI")
    assert KNOWN_TOOLS["Scrunch"] == ("Scrunch", "Scrunch AI")
    assert KNOWN_TOOLS["Semrush AI Toolkit"] == ("Semrush AI Toolkit", "Semrush")
    assert KNOWN_TOOLS["Ahrefs Brand Radar"] == ("Ahrefs Brand Radar", "Ahrefs")
    assert KNOWN_TOOLS["Sextant AEO"] == ("Sextant AEO", "Sextant", "sextantaeo.com")
    for canonical, spellings in KNOWN_TOOLS.items():
        assert spellings[0] == canonical


def test_known_tools_fold_to_one_name():
    assert normalise(" semrush ") == "Semrush AI Toolkit"
    assert normalise("sextantaeo.com") == "Sextant AEO"
    assert normalise("SEXTANT") == "Sextant AEO"
    assert normalise("Otterly.AI") == "Otterly"
    assert normalise("Brandwatch") == "Brandwatch"
    assert normalise("  Brand Radar Pro ") == "Brand Radar Pro"
    assert normalise("") == "" and normalise("   ") == ""


def test_known_tools_is_read_only():
    with pytest.raises(TypeError):
        KNOWN_TOOLS["Acme"] = ("Acme",)  # type: ignore[index]


def test_merged_table_adds_spellings_without_losing_defaults():
    t = AliasTable.default().merged({"Acme CRM": ["Acme"]})
    assert t.normalise("acme") == "Acme CRM" and t.normalise("Peec") == "Peec AI"
    assert t.spellings("Acme CRM") == ("Acme CRM", "Acme")
    assert AliasTable.default().normalise("acme") == "acme"


def test_merged_onto_a_known_tool_extends_its_spellings():
    t = AliasTable.default().merged({"Profound": ["tryprofound.com", "profound"]})
    assert t.spellings("Profound") == ("Profound", "tryprofound.com")
    assert t.normalise("TryProfound.com") == "Profound"


def test_a_spelling_claimed_by_two_tools_is_refused():
    with pytest.raises(ValueError, match=r"Semrush.*Semrush AI Toolkit"):
        AliasTable.default().merged({"Semrush One": ["Semrush"]})


def test_blank_spellings_are_refused():
    with pytest.raises(ValueError):
        AliasTable.default().merged({"Acme CRM": [" "]})
    with pytest.raises(ValueError):
        AliasTable.default().merged({" ": ["Acme"]})


def test_names_lists_every_spelling_and_spellings_of_unknown_is_empty():
    names = AliasTable.default().names()
    assert "Peec" in names and "sextantaeo.com" in names and "Profound" in names
    assert len(names) == len(set(n.lower() for n in names))
    assert AliasTable.default().spellings("Brandwatch") == ()


def test_a_table_built_by_hand_keeps_the_canonical_name_first():
    t = AliasTable({"Acme CRM": ("Acme", "acme crm")})
    assert t.spellings("Acme CRM") == ("Acme CRM", "Acme")
    assert t.normalise("ACME CRM") == "Acme CRM"


def test_alias_tables_are_not_hashable():
    with pytest.raises(TypeError):
        hash(AliasTable.default())
