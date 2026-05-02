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
