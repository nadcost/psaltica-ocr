from psaltica_ocr.ocr_validator import OcrValidator, validate_composition
from psaltica_ocr.symbol_map import SymbolEntry, load_symbol_map


def by_icon(icon: str) -> SymbolEntry:
    for symbol in load_symbol_map("config/symbol_map.json").symbols:
        if symbol.icon == icon:
            return symbol
    raise AssertionError(f"Missing test symbol: {icon}")


def codes(composition: str) -> list[str]:
    return [error.code for error in validate_composition(composition)]


def test_valid_segment_start_key_signature_and_cluster() -> None:
    key = by_icon("PaKey")
    base = by_icon("Oligon")
    klasma = by_icon("Klasma")
    isson = by_icon("Pa")

    errors = validate_composition(f"{key.insert}{base.insert}{klasma.variants['short']}{isson.variants['short']}")

    assert errors == []


def test_valid_mid_segment_key_signature() -> None:
    base = by_icon("Oligon")
    mid_key = by_icon("DhiKey(Down)")

    assert validate_composition(f"{base.insert}{mid_key.insert}{base.insert}") == []


def test_valid_atomic_rest() -> None:
    rest = by_icon("Siopi1")

    assert validate_composition(rest.insert or "") == []


def test_unknown_char_is_rejected() -> None:
    assert codes("\uffff") == ["unknown_char"]


def test_trailing_unknown_after_valid_cluster_is_rejected() -> None:
    base = by_icon("Oligon")

    assert codes(f"{base.insert}~") == ["unknown_char"]


def test_modifier_without_base_is_rejected() -> None:
    klasma = by_icon("Klasma")

    assert codes(klasma.variants["short"] or "") == ["modifier_without_base"]


def test_duplicate_modifier_on_same_base_is_rejected() -> None:
    base = by_icon("Oligon")
    klasma = by_icon("Klasma")

    assert codes(f"{base.insert}{klasma.variants['short']}{klasma.variants['long']}") == ["duplicate_modifier"]


def test_segment_start_key_signature_mid_segment_is_rejected() -> None:
    key = by_icon("PaKey")
    base = by_icon("Oligon")

    assert codes(f"{base.insert}{key.insert}") == ["key_signature_role"]


def test_mid_only_key_signature_at_segment_start_is_rejected() -> None:
    mid_key = by_icon("DhiKey(Down)")

    assert codes(mid_key.insert or "") == ["key_signature_role"]


def test_mapping_input_validates_segments() -> None:
    validator = OcrValidator.from_path("config/symbol_map.json")
    base = by_icon("Oligon")
    payload = {"segments": [{"composition": base.insert}, {"composition": "\uffff"}]}

    errors = validator.validate(payload)

    assert [(error.segmentIndex, error.position, error.code) for error in errors] == [
        (1, 0, "unknown_char")
    ]
