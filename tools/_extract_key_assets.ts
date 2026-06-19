// Extract key-signature GIF asset paths from the Psaltica Praxis app.
//
// Key signatures render in the app as GIF artwork, not font glyphs, so the
// font-based thumbnails in the OCR review UI don't match what the user sees.
// This emits {label, icon, asset} for every key signature so the OCR side can
// show the real artwork. Run with cwd = praxis root (see tools/sync_key_assets.py):
//   npx tsx tools/_extract_key_assets.ts --out <file>

import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const moduleUrl = (root: string, rel: string) => pathToFileURL(join(root, rel)).href;

type KeySignature = { label: string; icon: string; asset: string };

const main = async () => {
  const args = process.argv.slice(2);
  const outIndex = args.indexOf("--out");
  const outPath = outIndex >= 0 ? args[outIndex + 1] : "key_assets_raw.json";
  const praxisRoot = process.cwd();

  const keySignatures = await import(moduleUrl(praxisRoot, "app/core/keySignatures.ts"));
  const rows = (keySignatures.RAW_KEY_SIGNATURES as KeySignature[]).map((k) => ({
    label: k.label,
    icon: k.icon,
    asset: k.asset,
  }));

  writeFileSync(outPath, JSON.stringify(rows, null, 2) + "\n");
  console.error(`Wrote ${rows.length} key-signature assets -> ${outPath}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
