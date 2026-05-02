from pathlib import Path

from psaltica_ocr.symbol_map import load_symbol_map


SYMBOL_MAP_PATH = Path("config/symbol_map.json")


def test_symbol_map_loads() -> None:
    symbol_map = load_symbol_map(SYMBOL_MAP_PATH)

    assert symbol_map.meta.key_signature_count > 0
    assert symbol_map.meta.action_char_map_count > 0
    assert len(symbol_map.symbols) > symbol_map.meta.key_signature_count


def test_meta_counts_match_generated_content() -> None:
    symbol_map = load_symbol_map(SYMBOL_MAP_PATH)
    toolbar_backed = [symbol for symbol in symbol_map.symbols if symbol.key_id is None]
    raw_key_signatures = [symbol for symbol in symbol_map.symbols if symbol.group == "key_signature"]

    assert len(toolbar_backed) == sum(symbol_map.meta.toolbar_counts.values())
    assert len(raw_key_signatures) == symbol_map.meta.key_signature_count
    assert len(symbol_map.meta.action_icons) == symbol_map.meta.action_char_map_count


def test_chars_are_unique_within_group_and_variant() -> None:
    symbol_map = load_symbol_map(SYMBOL_MAP_PATH)
    seen: dict[tuple[str, str, str], str] = {}

    for symbol in symbol_map.symbols:
        for variant, char in symbol.variants.items():
            if not char:
                continue
            key = (symbol.group, variant, char)
            previous = seen.get(key)
            assert previous is None, f"{char!r} reused in {symbol.group}.{variant}: {previous}, {symbol.icon}"
            seen[key] = symbol.icon


def test_key_signature_inserts_decompose_to_known_chars() -> None:
    symbol_map = load_symbol_map(SYMBOL_MAP_PATH)
    known_chars = {
        char
        for symbol in symbol_map.symbols
        for char in [symbol.insert, *symbol.variants.values(), *symbol.react_chars.values(), *symbol.legacy_chars.values()]
        if char
    }
    known_single_chars = {char for value in known_chars for char in value}

    for symbol in symbol_map.symbols:
        if not symbol.is_key_signature:
            continue
        assert symbol.insert
        unknown = [char for char in symbol.insert if char not in known_single_chars]
        assert not unknown, f"{symbol.icon} has unknown chars: {unknown!r}"


def test_action_map_icons_are_represented_or_explicitly_orphaned() -> None:
    symbol_map = load_symbol_map(SYMBOL_MAP_PATH)
    represented = {symbol.icon for symbol in symbol_map.symbols}
    orphans = set(symbol_map.meta.orphan_action_icons)
    action_icons = set(symbol_map.meta.action_icons)

    assert represented.isdisjoint(orphans)
    assert symbol_map.meta.action_char_map_count == len(action_icons)
    assert action_icons <= represented | orphans
