// Free-model aggregator: one server, both wires, rating-ranked routing
// with a failover ladder across the free OpenRouter models, dialled
// through the grader gateway.
//
// Contract: listen on 0.0.0.0 at a free port, register with the env,
// answer POST /v1/chat/completions (OpenAI) and POST /v1/messages
// (Anthropic) in JSON and streaming forms, answer GET /v1/models in the
// wire its auth header names, poll GET /work and exit 0 on done. With
// --list, print the registered catalog and exit 0.
//
// Routing: every prompt starts at the top-rated model and walks down the
// ladder. A model that refuses (4xx/5xx, network failure, timeout) is
// skipped for the rest of the trial. An answer is served only when it
// carries the prompt token back; a successful answer without the token
// fails over to the next model, kept as a best-effort fallback. Upstream
// streams are never relayed: the gateway answer is buffered, checked,
// repaired where the gateway omitted data, then rendered in the client's
// wire, so the refusal trail header can be set before the body and a
// mid-stream failure can still fail over.
//
// Determinism: no wall clock and no randomness anywhere in routing or
// rendering. Ids come from a per-process counter, timestamps are fixed,
// and the only clock is a monotonic deadline that never fires when the
// gateway answers promptly, so a replay of the same requests reproduces
// the same answers.

import http from "node:http";
import { readFileSync } from "node:fs";
import {
  modelsFromBrief,
  buildCatalog,
  emptyCatalog,
  type Catalog,
  type RankedModel,
} from "./lib/providers.ts";
import {
  anthropicToChat,
  openaiToChat,
  normalizeUpstream,
  declaredToolNames,
  tokenInRequest,
  renderOpenAIAnswer,
  renderOpenAIStream,
  renderAnthropicAnswer,
  renderAnthropicStream,
  renderOpenAIModels,
  renderAnthropicModels,
  type NormalizedAnswer,
  type Wire,
} from "./lib/wire.ts";

interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

interface GatewayResult {
  thrown: boolean;
  timeout: boolean;
  status: number;
  json: unknown;
}

const MAX_BODY_BYTES = 4 * 1024 * 1024;
const MAX_ATTEMPTS_PER_PROMPT = 6;
const MAX_CALLS_PER_TRIAL = 48;
const PROMPT_DEADLINE_MS = 25000;
const ATTEMPT_TIMEOUTS_MS = [15000, 8000, 6000, 5000, 4000, 3000];
const TRAIL_HEADER = "x-makefun-upstream-failures";

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function loadBriefText(): string | null {
  const path = process.env.MAKEFUN_PROMPT_FILE;
  if (!path) return null;
  try {
    return readFileSync(path, "utf8");
  } catch {
    return null;
  }
}

function gatewayBase(): string {
  const base = process.env.OPENAI_BASE_URL || "http://llm:8080/v1";
  return base.replace(/\/$/, "");
}

function gatewayKey(): string {
  return process.env.OPENAI_API_KEY || "";
}

// The refusal code for one gateway answer: the string code the gateway or
// provider sent, else its error type, else the first 64 characters of its
// message, else an http_<status> placeholder. Always a string, at most 64
// characters, so the trail header stays well formed.
function refusalCode(status: number, json: unknown): string {
  if (isRecord(json) && isRecord(json.error)) {
    const err = json.error;
    if (typeof err.code === "string" && err.code) {
      return err.code.slice(0, 64);
    }
    if (typeof err.type === "string" && err.type) {
      return err.type.slice(0, 64);
    }
    if (typeof err.message === "string" && err.message) {
      return err.message.slice(0, 64);
    }
  }
  return `http_${status}`;
}

async function dialGateway(
  body: Record<string, unknown>,
  timeoutMs: number,
): Promise<GatewayResult> {
  const url = `${gatewayBase()}/chat/completions`;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${gatewayKey()}`,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    const text = await res.text();
    let json: unknown = null;
    if (text.length <= MAX_BODY_BYTES) {
      try {
        json = JSON.parse(text);
      } catch {
        json = null;
      }
    }
    return { thrown: false, timeout: false, status: res.status, json };
  } catch (err) {
    const name = err instanceof Error ? err.name : "";
    return {
      thrown: true,
      timeout: name === "TimeoutError" || name === "AbortError",
      status: 0,
      json: null,
    };
  }
}

class Aggregator {
  private ladder: RankedModel[];
  private refused = new Set<string>();
  private upstreamCalls = 0;
  private responseSeq = 0;

  constructor(models: RankedModel[]) {
    this.ladder = models;
  }

  private nextId(prefix: string): string {
    this.responseSeq += 1;
    return `${prefix}-fma02-${this.responseSeq}`;
  }

  // Serve one completion request. Returns the status, headers and body to
  // send, already rendered in the client's wire.
  async serve(
    wire: Wire,
    body: Record<string, unknown>,
  ): Promise<{ status: number; headers: Record<string, string>; payload: string }> {
    const stream = body.stream === true;
    const token = tokenInRequest(body);
    const declared = declaredToolNames(body, wire);
    const started = performance.now();
    const trail: TrailRow[] = [];
    let fallback: { answer: NormalizedAnswer; model: string } | null = null;
    let served: { answer: NormalizedAnswer; model: string } | null = null;

    // Unrefused models first in rating order, refused models only as a
    // last resort, never the same model twice for one prompt.
    const candidates = [
      ...this.ladder.filter((m) => !this.refused.has(m.id)),
      ...this.ladder.filter((m) => this.refused.has(m.id)),
    ];
    let attempts = 0;
    for (const model of candidates) {
      if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
      if (this.upstreamCalls >= MAX_CALLS_PER_TRIAL) break;
      const remaining = PROMPT_DEADLINE_MS - (performance.now() - started);
      if (remaining < 1500) break;
      const timeout = Math.min(
        ATTEMPT_TIMEOUTS_MS[attempts] ?? 3000,
        Math.max(1000, Math.floor(remaining) - 500),
      );
      attempts += 1;
      this.upstreamCalls += 1;
      const upstreamBody =
        wire === "anthropic"
          ? anthropicToChat(body, model.id)
          : openaiToChat(body, model.id);
      const res = await dialGateway(upstreamBody, timeout);
      if (res.thrown) {
        trail.push({
          status: res.timeout ? 504 : 502,
          code: res.timeout ? "gateway_timeout" : "network_error",
          model: model.id,
        });
        this.refused.add(model.id);
        continue;
      }
      if (res.status < 200 || res.status >= 300) {
        trail.push({
          status: res.status,
          code: refusalCode(res.status, res.json),
          model: model.id,
        });
        this.refused.add(model.id);
        continue;
      }
      const answer = normalizeUpstream(res.json, declared);
      if (answer === null) continue;
      if (token === null || answer.text.includes(token)) {
        served = { answer, model: model.id };
        break;
      }
      // A real answer that misses the token: keep it as the fallback and
      // fail over. The model itself is healthy, so it stays in the pool.
      if (fallback === null) fallback = { answer, model: model.id };
    }

    const headers: Record<string, string> = {};
    if (trail.length > 0) {
      headers[TRAIL_HEADER] = JSON.stringify(trail.slice(0, 8));
    }
    const chosen = served ?? fallback;
    if (chosen === null) {
      headers["content-type"] = "application/json";
      const payload =
        wire === "anthropic"
          ? JSON.stringify({
              type: "error",
              error: {
                type: "api_error",
                message: "all upstream models refused",
              },
            })
          : JSON.stringify({
              error: {
                message: "all upstream models refused",
                type: "server_error",
                code: "upstream_refused",
              },
            });
      return { status: 502, headers, payload };
    }
    if (wire === "openai") {
      if (stream) {
        headers["content-type"] = "text/event-stream";
        headers["cache-control"] = "no-cache";
        return {
          status: 200,
          headers,
          payload: renderOpenAIStream(
            chosen.answer,
            chosen.model,
            this.nextId("chatcmpl"),
          ),
        };
      }
      headers["content-type"] = "application/json";
      return {
        status: 200,
        headers,
        payload: JSON.stringify(
          renderOpenAIAnswer(
            chosen.answer,
            chosen.model,
            this.nextId("chatcmpl"),
          ),
        ),
      };
    }
    if (stream) {
      headers["content-type"] = "text/event-stream";
      headers["cache-control"] = "no-cache";
      return {
        status: 200,
        headers,
        payload: renderAnthropicStream(
          chosen.answer,
          chosen.model,
          this.nextId("msg"),
        ),
      };
    }
    headers["content-type"] = "application/json";
    return {
      status: 200,
      headers,
      payload: JSON.stringify(
        renderAnthropicAnswer(
          chosen.answer,
          chosen.model,
          this.nextId("msg"),
        ),
      ),
    };
  }
}

function readBody(req: http.IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    let tooBig = false;
    req.on("data", (chunk: Buffer) => {
      total += chunk.length;
      if (total > MAX_BODY_BYTES) {
        tooBig = true;
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (tooBig) {
        reject(new Error("body_too_large"));
        return;
      }
      resolve(Buffer.concat(chunks));
    });
    req.on("error", reject);
  });
}

function badRequest(wire: Wire, message: string): { status: number; payload: string } {
  if (wire === "anthropic") {
    return {
      status: 400,
      payload: JSON.stringify({
        type: "error",
        error: { type: "invalid_request_error", message },
      }),
    };
  }
  return {
    status: 400,
    payload: JSON.stringify({
      error: { message, type: "invalid_request_error", code: "invalid_json" },
    }),
  };
}

async function main(): Promise<void> {
  const briefText = loadBriefText();
  const models = briefText ? modelsFromBrief(briefText) : [];
  const catalog: Catalog =
    models.length > 0 ? buildCatalog(models) : emptyCatalog();

  if (process.argv.includes("--list")) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`, () => process.exit(0));
    return;
  }

  const aggregator = new Aggregator(models);
  const server = http.createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://local");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          // The wire is the auth header: x-api-key names Anthropic,
          // Authorization: Bearer names OpenAI.
          const wire: Wire =
            req.headers["x-api-key"] !== undefined ? "anthropic" : "openai";
          const payload =
            wire === "anthropic"
              ? renderAnthropicModels(models)
              : renderOpenAIModels(models);
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(payload));
          return;
        }
        if (
          req.method === "POST" &&
          (url.pathname === "/v1/chat/completions" ||
            url.pathname === "/v1/messages")
        ) {
          const wire: Wire =
            url.pathname === "/v1/messages" ? "anthropic" : "openai";
          let raw: Buffer;
          try {
            raw = await readBody(req);
          } catch {
            res.writeHead(413, { "content-type": "application/json" });
            res.end(
              JSON.stringify({
                error: {
                  message: "request body too large",
                  type: "invalid_request_error",
                  code: "request_too_large",
                },
              }),
            );
            return;
          }
          let body: unknown;
          try {
            body = JSON.parse(raw.toString("utf8"));
          } catch {
            const bad = badRequest(wire, "request body is not valid JSON");
            res.writeHead(bad.status, { "content-type": "application/json" });
            res.end(bad.payload);
            return;
          }
          if (!isRecord(body) || !Array.isArray(body.messages)) {
            const bad = badRequest(wire, "request body needs a messages array");
            res.writeHead(bad.status, { "content-type": "application/json" });
            res.end(bad.payload);
            return;
          }
          const result = await aggregator.serve(wire, body);
          res.writeHead(result.status, result.headers);
          res.end(result.payload);
          return;
        }
        res.writeHead(404, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            error: {
              message: "not found",
              type: "invalid_request_error",
              code: "not_found",
            },
          }),
        );
      } catch (err) {
        process.stderr.write(`handler error: ${String(err)}\n`);
        if (!res.headersSent) {
          res.writeHead(500, { "content-type": "application/json" });
        }
        res.end(
          JSON.stringify({
            error: {
              message: "internal error",
              type: "server_error",
              code: "internal_error",
            },
          }),
        );
      }
    })();
  });

  // A free port of our own: the env owns 8080.
  let port = 0;
  for (let attempt = 0; attempt < 3 && port === 0; attempt++) {
    port = await new Promise<number>((resolve) => {
      server.listen(0, "0.0.0.0", () => {
        const addr = server.address();
        const picked =
          addr !== null && typeof addr === "object" ? addr.port : 0;
        if (picked === 8080) {
          server.close(() => resolve(0));
          return;
        }
        resolve(picked);
      });
    });
  }
  if (port === 0) {
    process.stderr.write("could not bind a port\n");
    process.exit(1);
  }

  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (!envUrl) {
    process.stderr.write(`aggregator listening on ${port} (no env, serving only)\n`);
    return;
  }
  const base = envUrl.replace(/\/$/, "");
  let registered = false;
  let fatalStatus = 0;
  for (let i = 0; i < 300 && !registered && fatalStatus === 0; i++) {
    try {
      const res = await fetch(`${base}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        registered = true;
      } else if (
        res.status >= 400 &&
        res.status < 500 &&
        res.status !== 408 &&
        res.status !== 429
      ) {
        // A 4xx against a fixed register body is permanent: the env
        // rejects the payload itself, and retrying the same bytes for
        // a minute cannot change that. Fail fast and say so.
        fatalStatus = res.status;
      }
    } catch {
      // The env may not be up yet; retry on a fixed interval.
    }
    if (!registered && fatalStatus === 0) await sleep(200);
  }
  if (!registered) {
    process.stderr.write(
      fatalStatus > 0
        ? `registration rejected with HTTP ${fatalStatus}\n`
        : "registration failed\n",
    );
    process.exit(1);
  }
  while (true) {
    try {
      const res = await fetch(`${base}/work`, {
        signal: AbortSignal.timeout(5000),
      });
      const work: unknown = await res.json();
      if (isRecord(work) && work.done === true) {
        process.exit(0);
      }
    } catch {
      // Transient env failure; keep polling.
    }
    await sleep(100);
  }
}

void main();
