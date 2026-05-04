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

This writes `data/font_shape_groups.json`, `data/annotations/font_shape_matches.json`, and `data/annotations/font_shape_matches.csv`. Glyphs are grouped after tight ink cropping and centering, so x/y attachment-position variants collapse into the same shape group.

Tune `--shape-threshold` to control how aggressively glyphs are grouped, and `--match-threshold` to control page-match precision.
