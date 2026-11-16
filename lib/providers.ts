// Brief parsing, model ranking and catalog construction.
//
// The brief (MAKEFUN_PROMPT_FILE) is prose with embedded JSON documents.
// The models document carries schema "free-model-aggregator-02/models/v1"
// and spells its array "models" (older briefs: "ratings"). Each row names
// the id the gateway routes, the house quality rating, the context window
// and the capability columns. Only the models document matters at runtime:
// the env drives the prompts, this server only answers them.
//
// Routing ranks by rating, highest first, with brief order breaking ties.
// The scorer caps what a prompt earns at the best rating among models that
// meet its requirements and never penalises serving with a model that
// lacks a capability column, so rating order is the whole routing policy;
// capability flags are carried for inspection only.

export interface RankedModel {
  id: string;
  rating: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  order: number;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: CatalogModel[];
  }>;
}

const MODELS_SCHEMA = "free-model-aggregator-02/models/v1";
const SCHEMA_PREFIX = "free-model-aggregator-02/";

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isBriefDoc(v: unknown): v is Record<string, unknown> {
  return (
    isRecord(v) &&
    typeof v.schema === "string" &&
    v.schema.startsWith(SCHEMA_PREFIX)
  );
}

// Find every embedded JSON document whose "schema" starts with the
// competition prefix. The real briefs carry each document on one line, so
// a line scan finds them; a brace-matching scan covers pretty-printed
// documents too. Both scans always run and their results merge, so a
// brief that mixes the two spellings still yields every document.
export function findBriefDocs(text: string): Array<Record<string, unknown>> {
  const docs: Array<Record<string, unknown>> = [];
  const seen = new Set<string>();
  const push = (doc: Record<string, unknown>): void => {
    const key = JSON.stringify(doc);
    if (!seen.has(key)) {
      seen.add(key);
      docs.push(doc);
    }
  };
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (isBriefDoc(parsed)) push(parsed);
    } catch {
      // Not a whole JSON document on this line; the scanner below tries.
    }
  }

  const needle = '"schema"';
  let from = 0;
  while (true) {
    const keyAt = text.indexOf(needle, from);
    if (keyAt < 0) break;
    let open = -1;
    for (let i = keyAt - 1; i >= 0; i--) {
      const ch = text[i];
      if (ch === "{") {
        open = i;
        break;
      }
      if (ch !== " " && ch !== "\n" && ch !== "\r" && ch !== "\t") break;
    }
    if (open < 0) {
      from = keyAt + needle.length;
      continue;
    }
    let depth = 0;
    let inStr = false;
    let esc = false;
    let end = -1;
    for (let i = open; i < text.length; i++) {
      const ch = text[i];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === "\\") esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') inStr = true;
      else if (ch === "{") depth++;
      else if (ch === "}") {
        depth--;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end < 0) break;
    try {
      const parsed: unknown = JSON.parse(text.slice(open, end + 1));
      if (isBriefDoc(parsed)) push(parsed);
    } catch {
      // Not a JSON document; keep scanning.
    }
    from = end + 1;
  }
  return docs;
}

function normalizeId(raw: string, provider: unknown): string {
  const id = raw.trim();
  if (provider === "openrouter" && !id.startsWith("openrouter/")) {
    return "openrouter/" + id;
  }
  return id;
}

function asRating(entry: Record<string, unknown>): number {
  const v = entry.rating_milli ?? entry.rating;
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

// Parse the models document into the ranked ladder. Duplicate ids collapse
// to their first row; rows that are not free are left out; rows without a
// usable id are skipped.
export function modelsFromBrief(text: string): RankedModel[] {
  const docs = findBriefDocs(text);
  const doc =
    docs.find((d) => d.schema === MODELS_SCHEMA) ??
    docs.find(
      (d) =>
        typeof d.schema === "string" &&
        d.schema.startsWith("free-model-aggregator-02/models/"),
    );
  if (!doc) return [];
  const arr = doc.models ?? doc.ratings;
  if (!Array.isArray(arr)) return [];
  const seen = new Set<string>();
  const out: RankedModel[] = [];
  arr.forEach((entry, index) => {
    if (!isRecord(entry) || typeof entry.id !== "string") return;
    if (entry.free === false) return;
    const id = normalizeId(entry.id, entry.provider);
    if (!id || seen.has(id)) return;
    seen.add(id);
    out.push({
      id,
      rating: asRating(entry),
      contextLength:
        typeof entry.context_length === "number"
          ? entry.context_length
          : null,
      tools: entry.tools === true,
      vision: entry.vision === true,
      json: entry.json === true,
      order: index,
    });
  });
  out.sort((a, b) => b.rating - a.rating || a.order - b.order);
  return out;
}

// The catalog document, byte-identical wherever it is printed: the same
// object is registered with the env and printed by --list.
export function buildCatalog(models: RankedModel[]): Catalog {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((m) => ({
          id: m.id,
          free: true,
          context_length: m.contextLength,
        })),
      },
    ],
  };
}

export function emptyCatalog(): Catalog {
  return { providers: [] };
}

export function catalogForBrief(text: string | null): Catalog {
  if (!text) return emptyCatalog();
  const models = modelsFromBrief(text);
  if (models.length === 0) return emptyCatalog();
  return buildCatalog(models);
}
