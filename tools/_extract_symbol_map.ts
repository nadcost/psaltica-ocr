import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

type ToolbarItem = {
  icon: string;
  insert?: string;
  short?: string;
  long?: string;
  extraShort?: string;
  under?: string;
};

type ActionCharInfo = {
  icon: string;
  legacy: Record<string, string[]>;
  react: Record<string, string[]>;
};

type KeySignature = {
  keyId: number;
  label: string;
  icon: string;
  category: string;
  role: string;
  insert: string;
  basePitch: number | null;
};

type BaseCharInfo = {
  icon: string;
  length: "long" | "short" | "extraShort" | "normal";
  heavyTop: boolean;
  klasmaPlacement: "topCenter" | "topLeft" | "topRight" | "bottomCenter";
};

type SymbolEntry = {
  icon: string;
  label: string;
  group:
    | "neume"
    | "gorgon"
    | "modulation"
    | "isson"
    | "mode"
    | "key_signature"
    | "rest"
    | "ornament";
  role: "base" | "modifier" | "key_signature" | "rest" | "ornament";
  variants: Record<string, string | null>;
  insert: string | null;
  isBase: boolean;
  isModifier: boolean;
  isKeySignature: boolean;
  keyId: number | null;
  keySignatureRole: string | null;
  category: string | null;
  basePitch: number | null;
  length: BaseCharInfo["length"] | null;
  heavyTop: boolean;
  klasmaPlacement: BaseCharInfo["klasmaPlacement"] | null;
  legacyChars: Record<string, string>;
  reactChars: Record<string, string>;
};

const variantKeys = ["insert", "short", "long", "extraShort", "under"] as const;

const parseArgs = () => {
  const outIndex = process.argv.indexOf("--out");
  if (outIndex === -1 || !process.argv[outIndex + 1]) {
    throw new Error("Usage: _extract_symbol_map.ts --out <path>");
  }
  return { outPath: process.argv[outIndex + 1] };
};

const moduleUrl = (praxisRoot: string, relativePath: string) =>
  pathToFileURL(join(praxisRoot, relativePath)).href;

const firstChars = (chars: Record<string, string[]> | undefined) => {
  const result: Record<string, string> = {};
  Object.entries(chars ?? {}).forEach(([variant, values]) => {
    const first = values.find((value) => typeof value === "string" && value.length > 0);
    if (first) result[variant] = first;
  });
  return result;
};

const variantsFromToolbar = (item: ToolbarItem) => {
  const variants: Record<string, string | null> = {};
  variantKeys.forEach((key) => {
    variants[key] = item[key] ?? null;
  });
  return variants;
};

const variantsFromInsert = (insert: string) => ({
  insert,
  short: null,
  long: null,
  extraShort: null,
  under: null,
});

const primaryInsert = (variants: Record<string, string | null>) =>
  variants.insert ?? variants.short ?? variants.long ?? variants.extraShort ?? variants.under ?? null;

const classGroupForNeume = (item: ToolbarItem) => {
  if (String(item.icon).startsWith("Siopi")) return "rest";
  return "neume";
};

const roleForGroup = (group: SymbolEntry["group"]): SymbolEntry["role"] => {
  if (group === "neume") return "base";
  if (group === "mode" || group === "key_signature") return "key_signature";
  if (group === "rest") return "rest";
  if (group === "ornament") return "ornament";
  return "modifier";
};

const makeToolbarEntry = (
  item: ToolbarItem,
  group: SymbolEntry["group"],
  actionByIcon: Map<string, ActionCharInfo>,
  baseInfoByIcon: Map<string, BaseCharInfo>,
): SymbolEntry => {
  const icon = String(item.icon);
  const variants = variantsFromToolbar(item);
  const role = roleForGroup(group);
  const action = actionByIcon.get(icon);
  const baseInfo = baseInfoByIcon.get(icon);

  return {
    icon,
    label: icon,
    group,
    role,
    variants,
    insert: primaryInsert(variants),
    isBase: role === "base",
    isModifier: role === "modifier",
    isKeySignature: role === "key_signature",
    keyId: null,
    keySignatureRole: group === "mode" ? "segmentStart" : null,
    category: null,
    basePitch: null,
    length: baseInfo?.length ?? null,
    heavyTop: baseInfo?.heavyTop ?? false,
    klasmaPlacement: baseInfo?.klasmaPlacement ?? null,
    legacyChars: firstChars(action?.legacy),
    reactChars: firstChars(action?.react),
  };
};

const makeKeySignatureEntry = (
  keySignature: KeySignature,
  actionByIcon: Map<string, ActionCharInfo>,
): SymbolEntry => {
  const action = actionByIcon.get(keySignature.icon);
  return {
    icon: keySignature.icon,
    label: keySignature.label,
    group: "key_signature",
    role: "key_signature",
    variants: variantsFromInsert(keySignature.insert),
    insert: keySignature.insert,
    isBase: false,
    isModifier: false,
    isKeySignature: true,
    keyId: keySignature.keyId,
    keySignatureRole: keySignature.role,
    category: keySignature.category,
    basePitch: keySignature.basePitch,
    length: null,
    heavyTop: false,
    klasmaPlacement: null,
    legacyChars: firstChars(action?.legacy),
    reactChars: firstChars(action?.react),
  };
};

const assertUniqueToolbarMembership = (
  toolbarMembership: Map<string, string[]>,
  keySignatureIcons: Set<string>,
) => {
  const invalid = [...toolbarMembership.entries()].filter(
    ([icon, groups]) => groups.length !== 1 && !keySignatureIcons.has(icon),
  );
  if (invalid.length > 0) {
    throw new Error(
      `Toolbar icons must appear in exactly one toolbar unless they are key signatures: ${invalid
        .map(([icon, groups]) => `${icon}=${groups.join(",")}`)
        .join("; ")}`,
    );
  }
};

const main = async () => {
  const { outPath } = parseArgs();
  const praxisRoot = process.cwd();

  const toolbars = await import(moduleUrl(praxisRoot, "app/core/toolbars.ts"));
  const keySignatures = await import(moduleUrl(praxisRoot, "app/core/keySignatures.ts"));
  const actionMap = await import(moduleUrl(praxisRoot, "app/core/music/actionMap.ts"));
  const clusterCatalog = await import(moduleUrl(praxisRoot, "app/core/notation/clusterCatalog.ts"));

  const actionByIcon = new Map<string, ActionCharInfo>(
    (actionMap.ACTION_CHAR_MAP as ActionCharInfo[]).map((entry) => [entry.icon, entry]),
  );

  const baseInfoByIcon = new Map<string, BaseCharInfo>();
  for (const info of (clusterCatalog.BASE_CHAR_TO_INFO as Map<string, BaseCharInfo>).values()) {
    baseInfoByIcon.set(info.icon, info);
  }

  const rawKeySignatures = keySignatures.RAW_KEY_SIGNATURES as KeySignature[];
  const keySignatureIcons = new Set(rawKeySignatures.map((entry) => entry.icon));

  const toolbarEntries: Array<[SymbolEntry["group"], ToolbarItem[]]> = [
    ["mode", toolbars.modesToolbar],
    ["modulation", toolbars.modulationToolbar],
    ["neume", toolbars.neumeToolbar],
    ["gorgon", toolbars.gorgonToolbar],
    ["isson", toolbars.issonToolbar],
  ];

  const toolbarMembership = new Map<string, string[]>();
  for (const [group, items] of toolbarEntries) {
    for (const item of items as ToolbarItem[]) {
      const icon = String(item.icon);
      const groups = toolbarMembership.get(icon) ?? [];
      groups.push(group);
      toolbarMembership.set(icon, groups);
    }
  }
  assertUniqueToolbarMembership(toolbarMembership, keySignatureIcons);

  const symbols: SymbolEntry[] = [];
  for (const [group, items] of toolbarEntries) {
    for (const item of items as ToolbarItem[]) {
      const resolvedGroup = group === "neume" ? classGroupForNeume(item) : group;
      symbols.push(makeToolbarEntry(item, resolvedGroup, actionByIcon, baseInfoByIcon));
    }
  }
  symbols.push(...rawKeySignatures.map((entry) => makeKeySignatureEntry(entry, actionByIcon)));

  const representedActionIcons = new Set([
    ...toolbarMembership.keys(),
    ...keySignatureIcons,
  ]);
  const orphanActionIcons = (actionMap.ACTION_CHAR_MAP as ActionCharInfo[])
    .map((entry) => entry.icon)
    .filter((icon) => !representedActionIcons.has(icon))
    .sort();
  const actionIcons = (actionMap.ACTION_CHAR_MAP as ActionCharInfo[])
    .map((entry) => entry.icon)
    .sort();

  const output = {
    _meta: {
      generatedBy: "tools/_extract_symbol_map.ts",
      praxisRoot,
      toolbarCounts: {
        modesToolbar: toolbars.modesToolbar.length,
        modulationToolbar: toolbars.modulationToolbar.length,
        neumeToolbar: toolbars.neumeToolbar.length,
        gorgonToolbar: toolbars.gorgonToolbar.length,
        issonToolbar: toolbars.issonToolbar.length,
      },
      keySignatureCount: rawKeySignatures.length,
      actionCharMapCount: actionMap.ACTION_CHAR_MAP.length,
      actionIcons,
      reactSequenceCount: Object.keys(actionMap.REACT_SEQUENCE_TO_ACTION).length,
      legacySequenceCount: Object.keys(actionMap.LEGACY_SEQUENCE_TO_ACTION).length,
      allSequenceCount: Object.keys(actionMap.ALL_SEQUENCE_TO_ACTION).length,
      orphanActionIcons,
    },
    symbols,
  };

  writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
