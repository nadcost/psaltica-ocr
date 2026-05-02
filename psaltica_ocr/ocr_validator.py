"""Strict validation for OCR composition output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from psaltica_ocr.symbol_map import SymbolEntry, SymbolMap, load_symbol_map


@dataclass(frozen=True)
class ValidationError:
    segmentIndex: int
    position: int
    code: str
    message: str


@dataclass(frozen=True)
class ModifierInfo:
    icon: str
    group: str
    variant: str


class OcrValidator:
    """Validate composition strings against generated Psaltica symbol metadata."""

    def __init__(self, symbol_map: SymbolMap) -> None:
        self.symbol_map = symbol_map
        self.base_chars = self._base_chars(symbol_map.symbols)
        self.modifier_chars = self._modifier_chars(symbol_map.symbols)
        self.atomic_chars = self._atomic_chars(symbol_map.symbols)
        self.key_signatures = self._key_signatures(symbol_map.symbols)

    @classmethod
    def from_path(cls, path: str | Path = "config/symbol_map.json") -> "OcrValidator":
        return cls(load_symbol_map(path))

    def validate(self, composition: str | Sequence[str] | Mapping[str, Any]) -> list[ValidationError]:
        segments = normalize_segments(composition)
        errors: list[ValidationError] = []
        for segment_index, segment in enumerate(segments):
            errors.extend(self._validate_segment(segment, segment_index))
        return errors

    def _validate_segment(self, segment: str, segment_index: int) -> list[ValidationError]:
        errors: list[ValidationError] = []
        position = 0
        while position < len(segment):
            key_signature = self._match_key_signature(segment, position)
            if key_signature is not None:
                errors.extend(self._validate_key_signature_role(key_signature, segment_index, position))
                position += len(key_signature.insert or "")
                continue

            char = segment[position]
            if char in self.base_chars:
                position = self._consume_base_cluster(segment, segment_index, position, errors)
                continue

            if char in self.atomic_chars:
                position += 1
                continue

            if char in self.modifier_chars:
                errors.append(
                    ValidationError(
                        segment_index,
                        position,
                        "modifier_without_base",
                        f"Modifier {self.modifier_chars[char].icon} is not attached to a base neume.",
                    ),
                )
                position += 1
                continue

            errors.append(
                ValidationError(
                    segment_index,
                    position,
                    "unknown_char",
                    f"Unknown OCR composition character {char!r}.",
                ),
            )
            position += 1
        return errors

    def _consume_base_cluster(
        self,
        segment: str,
        segment_index: int,
        position: int,
        errors: list[ValidationError],
    ) -> int:
        position += 1
        seen_modifiers: set[tuple[str, str]] = set()
        while position < len(segment):
            char = segment[position]
            modifier = self.modifier_chars.get(char)
            if modifier is None:
                break
            modifier_key = (modifier.group, modifier.icon)
            if modifier_key in seen_modifiers:
                errors.append(
                    ValidationError(
                        segment_index,
                        position,
                        "duplicate_modifier",
                        f"Duplicate modifier {modifier.icon} on the same base neume.",
                    ),
                )
            seen_modifiers.add(modifier_key)
            position += 1
        return position

    def _match_key_signature(self, segment: str, position: int) -> SymbolEntry | None:
        for key_signature in self.key_signatures:
            insert = key_signature.insert
            if insert and segment.startswith(insert, position):
                return key_signature
        return None

    def _validate_key_signature_role(
        self,
        key_signature: SymbolEntry,
        segment_index: int,
        position: int,
    ) -> list[ValidationError]:
        role = key_signature.key_signature_role
        if role == "segmentStart" and position != 0:
            return [
                ValidationError(
                    segment_index,
                    position,
                    "key_signature_role",
                    f"Key signature {key_signature.icon} is segment-start only.",
                )
            ]
        if role == "midOnly" and position == 0:
            return [
                ValidationError(
                    segment_index,
                    position,
                    "key_signature_role",
                    f"Key signature {key_signature.icon} is mid-segment only.",
                )
            ]
        return []

    @staticmethod
    def _base_chars(symbols: Iterable[SymbolEntry]) -> dict[str, SymbolEntry]:
        return {symbol.insert: symbol for symbol in symbols if symbol.is_base and symbol.insert}

    @staticmethod
    def _modifier_chars(symbols: Iterable[SymbolEntry]) -> dict[str, ModifierInfo]:
        modifiers: dict[str, ModifierInfo] = {}
        for symbol in symbols:
            if not symbol.is_modifier:
                continue
            for variant, char in symbol.variants.items():
                if char:
                    modifiers[char] = ModifierInfo(symbol.icon, symbol.group, variant)
        return modifiers

    @staticmethod
    def _atomic_chars(symbols: Iterable[SymbolEntry]) -> dict[str, SymbolEntry]:
        return {
            symbol.insert: symbol
            for symbol in symbols
            if symbol.role in {"rest", "ornament"} and symbol.insert and len(symbol.insert) == 1
        }

    @staticmethod
    def _key_signatures(symbols: Iterable[SymbolEntry]) -> list[SymbolEntry]:
        key_signatures = [
            symbol
            for symbol in symbols
            if symbol.is_key_signature and symbol.insert
        ]
        return sorted(key_signatures, key=lambda symbol: len(symbol.insert or ""), reverse=True)


def normalize_segments(composition: str | Sequence[str] | Mapping[str, Any]) -> list[str]:
    if isinstance(composition, str):
        return [composition]

    if isinstance(composition, Mapping):
        raw_segments = composition.get("segments", [])
        return [
            str(segment.get("composition", ""))
            for segment in raw_segments
            if isinstance(segment, Mapping)
        ]

    return [str(segment) for segment in composition]


def validate_composition(
    composition: str | Sequence[str] | Mapping[str, Any],
    symbol_map_path: str | Path = "config/symbol_map.json",
) -> list[ValidationError]:
    return OcrValidator.from_path(symbol_map_path).validate(composition)
