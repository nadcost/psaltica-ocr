# Psaltica OCR

Local OMR pipeline for printed Byzantine chant scores. The project is designed to recognize visual glyphs from clean printed PDFs and emit composition JSON compatible with Psaltica Praxis.

## Development

This is a uv-managed Python 3.11+ project.

```bash
uv sync --dev
uv run pytest
```

## System Dependencies

Install these before using the PDF rendering and symbol-map extraction tools:

```bash
brew install poppler node
```

- `poppler` provides PDF rendering support used by `pdf2image`.
- `node` is needed by the TypeScript symbol-map extractor that reads canonical symbols from Psaltica Praxis.

## Font Shape Matching

To group visually identical font glyphs and find those shape groups on rendered page images:

```bash
uv run tools/match_font_shape_groups.py --book Mass --pages-per-book 5
```

This writes `data/font_shape_groups.json`, `data/annotations/font_shape_matches.json`, `data/annotations/font_shape_matches.csv`, and `data/annotations/font_shape_matches.html`. Glyphs are grouped after tight ink cropping and centering, so x/y attachment-position variants collapse into the same shape group. The HTML report shows the matched representative glyph, every Unicode codepoint in the shape family, Psaltica app names when available, and detection frequency.

Tune `--shape-threshold` to control how aggressively glyphs are grouped, and `--match-threshold` to control page-match precision.
The matcher runs complex shapes first so larger composite glyphs suppress smaller component matches in the same region. It also applies built-in per-app tuning for `Apostrofos`, `Isson2`, and `Oligon`; use `--icon-threshold Name=0.80`, `--score-only-nms`, `--no-complex-first`, or `--no-icon-size-filters` when auditing alternate thresholds.
Known decorated variants of `OnePlusOneUp` are merged into the same shape family by default; add more with `--family-alias U+AAAA,U+BBBB` or disable built-in aliases with `--no-default-family-aliases`.
