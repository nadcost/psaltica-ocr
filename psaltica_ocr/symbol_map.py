"""Load and validate Psaltica OCR symbol-map artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


SymbolGroup = Literal[
    "neume",
    "gorgon",
    "modulation",
    "isson",
    "mode",
    "key_signature",
    "rest",
    "ornament",
]
SymbolRole = Literal["base", "modifier", "key_signature", "rest", "ornament"]
BaseLength = Literal["long", "short", "extraShort", "normal"]
KlasmaPlacement = Literal["topCenter", "topLeft", "topRight", "bottomCenter"]


class SymbolMapMeta(BaseModel):
    generated_by: str = Field(alias="generatedBy")
    praxis_root: str = Field(alias="praxisRoot")
    toolbar_counts: dict[str, int] = Field(alias="toolbarCounts")
    key_signature_count: int = Field(alias="keySignatureCount")
    action_char_map_count: int = Field(alias="actionCharMapCount")
    action_icons: list[str] = Field(alias="actionIcons")
    react_sequence_count: int = Field(alias="reactSequenceCount")
    legacy_sequence_count: int = Field(alias="legacySequenceCount")
    all_sequence_count: int = Field(alias="allSequenceCount")
    orphan_action_icons: list[str] = Field(alias="orphanActionIcons")


class SymbolEntry(BaseModel):
    icon: str
    label: str
    group: SymbolGroup
    role: SymbolRole
    variants: dict[str, str | None]
    insert: str | None
    is_base: bool = Field(alias="isBase")
    is_modifier: bool = Field(alias="isModifier")
    is_key_signature: bool = Field(alias="isKeySignature")
    key_id: int | None = Field(alias="keyId")
    category: str | None
    base_pitch: int | None = Field(alias="basePitch")
    length: BaseLength | None
    heavy_top: bool = Field(alias="heavyTop")
    klasma_placement: KlasmaPlacement | None = Field(alias="klasmaPlacement")
    legacy_chars: dict[str, str] = Field(alias="legacyChars")
    react_chars: dict[str, str] = Field(alias="reactChars")


class SymbolMap(BaseModel):
    meta: SymbolMapMeta = Field(alias="_meta")
    symbols: list[SymbolEntry]


def load_symbol_map(path: str | Path = "config/symbol_map.json") -> SymbolMap:
    """Load a generated symbol map from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return SymbolMap.model_validate(json.load(handle))
