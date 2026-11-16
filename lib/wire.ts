// Wire translation and answer rendering.
//
// The client speaks two wires: OpenAI Chat Completions and Anthropic
// Messages. Upstream is always OpenAI-shaped Chat Completions through the
// gateway, never Messages: an Anthropic request is translated into a Chat
// Completions body first, and the gateway answer is translated back into
// the wire the client used, streamed or not as it asked.
//
// Every answer and every streamed chunk carries the id that was dialled,
// never the bare slug the provider echoes back. Usage and stop reason are
// repaired only when the gateway omits them; real gateway values are
// never overwritten. Tool calls are filtered to the names the client
// declared, and dropped entirely when the client declared none.

export type Wire = "openai" | "anthropic";

export interface ToolCall {
  id: string;
  name: string;
  args: string;
}

export interface NormalizedAnswer {
  text: string;
  toolCalls: ToolCall[];
  finishReason: string | null;
  promptTokens: number | null;
  completionTokens: number | null;
}

export interface RepairedUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export const FIXED_CREATED = 1767225600;
export const FIXED_CREATED_AT = "2026-01-01T00:00:00Z";
export const MAX_TOKENS_CAP = 8192;

const TOKEN_RE = /\b[A-Z]{2,16}-[0-9]{4}\b/;

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function asText(v: unknown): string {
  return typeof v === "string" ? v : "";
}

// Collect every string the client sent (system prompt, message content,
// tool results) so the prompt token can be found in the request.
export function requestText(body: Record<string, unknown>): string {
  const parts: string[] = [];
  const walk = (v: unknown): void => {
    if (typeof v === "string") {
      parts.push(v);
      return;
    }
    if (Array.isArray(v)) {
      for (const item of v) walk(item);
      return;
    }
    if (isRecord(v)) {
      // Tool schemas and descriptions never carry the token; the token
      // lives in message content and the system prompt. Walk content-like
      // keys only, so a stray id in a tool schema cannot false-positive.
      for (const key of ["content", "system", "text"]) {
        if (key in v) walk(v[key]);
      }
      if (Array.isArray(v.messages)) walk(v.messages);
    }
  };
  walk(body);
  return parts.join("\n");
}

export function extractToken(text: string): string | null {
  const m = TOKEN_RE.exec(text);
  return m ? m[0] : null;
}

export function tokenInRequest(body: Record<string, unknown>): string | null {
  return extractToken(requestText(body));
}

function capMaxTokens(v: unknown, fallback: number): number {
  if (typeof v === "number" && Number.isFinite(v) && v >= 1) {
    return Math.min(Math.floor(v), MAX_TOKENS_CAP);
  }
  return fallback;
}

// Names the client declared. OpenAI tools carry function.name; Anthropic
// tools carry name.
export function declaredToolNames(
  body: Record<string, unknown>,
  wire: Wire,
): Set<string> {
  const names = new Set<string>();
  if (!Array.isArray(body.tools)) return names;
  for (const tool of body.tools) {
    if (!isRecord(tool)) continue;
    if (wire === "openai") {
      const fn = tool.function;
      if (isRecord(fn) && typeof fn.name === "string") names.add(fn.name);
    } else if (typeof tool.name === "string") {
      names.add(tool.name);
    }
  }
  return names;
}

function anthropicSystemMessages(
  system: unknown,
): Array<Record<string, unknown>> {
  if (typeof system === "string" && system) {
    return [{ role: "system", content: system }];
  }
  if (Array.isArray(system)) {
    const text = system
      .filter(isRecord)
      .filter((b) => b.type === "text")
      .map((b) => asText(b.text))
      .join("");
    if (text) return [{ role: "system", content: text }];
  }
  return [];
}

function blockText(block: Record<string, unknown>): string {
  if (block.type === "text") return asText(block.text);
  return "";
}

function stringifyToolContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter(isRecord)
      .map((b) => (b.type === "text" ? asText(b.text) : JSON.stringify(b)))
      .join("");
  }
  if (content === undefined || content === null) return "";
  return JSON.stringify(content);
}

// Translate an Anthropic Messages body into a Chat Completions body.
export function anthropicToChat(
  body: Record<string, unknown>,
  model: string,
): Record<string, unknown> {
  const messages: Array<Record<string, unknown>> = [];
  messages.push(...anthropicSystemMessages(body.system));
  const input = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of input) {
    if (!isRecord(msg)) continue;
    const role = msg.role === "assistant" ? "assistant" : "user";
    const content = msg.content;
    if (typeof content === "string") {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) {
      messages.push({ role, content: "" });
      continue;
    }
    const textParts: string[] = [];
    const imageParts: Array<Record<string, unknown>> = [];
    const toolCalls: Array<Record<string, unknown>> = [];
    const toolResults: Array<Record<string, unknown>> = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "text") {
        textParts.push(asText(block.text));
      } else if (block.type === "image" && isRecord(block.source)) {
        const src = block.source;
        if (src.type === "base64" && typeof src.data === "string") {
          const media = typeof src.media_type === "string" ? src.media_type : "image/png";
          imageParts.push({
            type: "image_url",
            image_url: { url: `data:${media};base64,${src.data}` },
          });
        }
      } else if (block.type === "tool_use") {
        toolCalls.push({
          id: asText(block.id) || `call_${toolCalls.length}`,
          type: "function",
          function: {
            name: asText(block.name),
            arguments: JSON.stringify(block.input ?? {}),
          },
        });
      } else if (block.type === "tool_result") {
        toolResults.push({
          role: "tool",
          tool_call_id: asText(block.tool_use_id),
          content: stringifyToolContent(block.content),
        });
      }
    }
    if (toolResults.length > 0) {
      // Tool results are their own messages; any text travels with the
      // first result's neighbour as a user message.
      if (textParts.join("") !== "") {
        messages.push({ role, content: textParts.join("") });
      }
      messages.push(...toolResults);
      continue;
    }
    const text = textParts.join("");
    if (imageParts.length > 0) {
      const parts: Array<Record<string, unknown>> = [];
      if (text !== "") parts.push({ type: "text", text });
      parts.push(...imageParts);
      const out: Record<string, unknown> = { role, content: parts };
      if (toolCalls.length > 0) out.tool_calls = toolCalls;
      messages.push(out);
    } else {
      const out: Record<string, unknown> = { role, content: text };
      if (toolCalls.length > 0) out.tool_calls = toolCalls;
      messages.push(out);
    }
  }
  const out: Record<string, unknown> = {
    model,
    messages,
    max_tokens: capMaxTokens(body.max_tokens, 1024),
    stream: false,
  };
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.stop)) out.stop = body.stop;
  if (Array.isArray(body.tools)) {
    out.tools = body.tools.filter(isRecord).map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: asText(tool.description),
        parameters: isRecord(tool.input_schema) ? tool.input_schema : {},
      },
    }));
  }
  if (body.tool_choice !== undefined) {
    out.tool_choice = anthropicToolChoice(body.tool_choice);
  }
  return out;
}

function anthropicToolChoice(choice: unknown): unknown {
  if (typeof choice === "string") {
    if (choice === "any") return "required";
    if (choice === "none") return "none";
    return "auto";
  }
  if (isRecord(choice)) {
    if (choice.type === "any") return "required";
    if (choice.type === "none") return "none";
    if (choice.type === "tool" && typeof choice.name === "string") {
      return { type: "function", function: { name: choice.name } };
    }
    return "auto";
  }
  return "auto";
}

// Pass an OpenAI body through, replacing the model and forcing a single
// buffered (non-streaming) upstream call. Only known fields cross; the
// gateway requires max_tokens, so one is always present.
export function openaiToChat(
  body: Record<string, unknown>,
  model: string,
): Record<string, unknown> {
  const maxTokens = capMaxTokens(
    body.max_tokens ?? body.max_completion_tokens ?? body.max_output_tokens,
    1024,
  );
  const out: Record<string, unknown> = {
    model,
    messages: Array.isArray(body.messages) ? body.messages : [],
    max_tokens: maxTokens,
    stream: false,
  };
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (body.stop !== undefined) out.stop = body.stop;
  if (Array.isArray(body.tools)) out.tools = body.tools;
  if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
  if (isRecord(body.response_format)) out.response_format = body.response_format;
  return out;
}

// Normalize one gateway answer. Returns null when the body carries no
// usable choice at all. Tool calls are filtered to declared names here,
// so no undeclared name can ever reach the client.
export function normalizeUpstream(
  json: unknown,
  declared: Set<string>,
): NormalizedAnswer | null {
  if (!isRecord(json)) return null;
  if (!Array.isArray(json.choices) || json.choices.length === 0) return null;
  const choice = json.choices[0];
  if (!isRecord(choice)) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  let text = "";
  const content = message.content;
  if (typeof content === "string") {
    text = content;
  } else if (Array.isArray(content)) {
    text = content.filter(isRecord).map(blockText).join("");
  }
  const toolCalls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      const name = asText(call.function.name);
      if (!name || !declared.has(name)) continue;
      const rawArgs = call.function.arguments;
      toolCalls.push({
        id: asText(call.id) || `call_${toolCalls.length}`,
        name,
        args: typeof rawArgs === "string" ? rawArgs : JSON.stringify(rawArgs ?? {}),
      });
    }
  }
  const usage = isRecord(json.usage) ? json.usage : {};
  return {
    text,
    toolCalls,
    finishReason:
      typeof choice.finish_reason === "string" ? choice.finish_reason : null,
    promptTokens:
      typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : null,
    completionTokens:
      typeof usage.completion_tokens === "number"
        ? usage.completion_tokens
        : null,
  };
}

// Repair usage only when the gateway omits it: input at least 0, output
// at least 1, estimated from the answer length when nothing real exists.
export function repairedUsage(ans: NormalizedAnswer): RepairedUsage {
  const prompt =
    ans.promptTokens !== null && ans.promptTokens >= 0
      ? Math.floor(ans.promptTokens)
      : 0;
  const completion =
    ans.completionTokens !== null && ans.completionTokens >= 1
      ? Math.floor(ans.completionTokens)
      : Math.max(1, Math.ceil(ans.text.length / 4));
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
  };
}

const OPENAI_FINISH = new Set([
  "stop",
  "length",
  "tool_calls",
  "function_call",
  "content_filter",
]);

export function openaiFinishReason(ans: NormalizedAnswer): string {
  if (ans.finishReason !== null && OPENAI_FINISH.has(ans.finishReason)) {
    // The filter may have removed every tool call the gateway sent; a
    // finish reason of tool_calls would then describe an answer that
    // carries none, so it downgrades to stop. A surviving call keeps it.
    if (
      (ans.finishReason === "tool_calls" ||
        ans.finishReason === "function_call") &&
      ans.toolCalls.length === 0
    ) {
      return "stop";
    }
    return ans.finishReason;
  }
  if (ans.toolCalls.length > 0) return "tool_calls";
  return "stop";
}

export function anthropicStopReason(ans: NormalizedAnswer): string {
  let mapped: string;
  switch (ans.finishReason) {
    case "stop":
      mapped = "end_turn";
      break;
    case "length":
      mapped = "max_tokens";
      break;
    case "tool_calls":
    case "function_call":
      mapped = "tool_use";
      break;
    case "content_filter":
      mapped = "refusal";
      break;
    default:
      mapped = ans.toolCalls.length > 0 ? "tool_use" : "end_turn";
  }
  if (mapped === "tool_use" && ans.toolCalls.length === 0) {
    return "end_turn";
  }
  return mapped;
}

export function renderOpenAIAnswer(
  ans: NormalizedAnswer,
  model: string,
  id: string,
): Record<string, unknown> {
  const usage = repairedUsage(ans);
  const message: Record<string, unknown> = {
    role: "assistant",
    content: ans.text,
  };
  if (ans.toolCalls.length > 0) {
    message.tool_calls = ans.toolCalls.map((c) => ({
      id: c.id,
      type: "function",
      function: { name: c.name, arguments: c.args },
    }));
  }
  return {
    id,
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: openaiFinishReason(ans),
      },
    ],
    usage,
  };
}

function splitPieces(text: string, size: number): string[] {
  if (text === "") return [];
  const pieces: string[] = [];
  for (let i = 0; i < text.length; i += size) {
    pieces.push(text.slice(i, i + size));
  }
  return pieces;
}

export function renderOpenAIStream(
  ans: NormalizedAnswer,
  model: string,
  id: string,
): string {
  const usage = repairedUsage(ans);
  const finish = openaiFinishReason(ans);
  const chunks: Array<Record<string, unknown>> = [];
  chunks.push({
    id,
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [
      { index: 0, delta: { role: "assistant", content: "" }, finish_reason: null },
    ],
  });
  for (const piece of splitPieces(ans.text, 120)) {
    chunks.push({
      id,
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [
        { index: 0, delta: { content: piece }, finish_reason: null },
      ],
    });
  }
  ans.toolCalls.forEach((call, index) => {
    chunks.push({
      id,
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index,
                id: call.id,
                type: "function",
                function: { name: call.name, arguments: call.args },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    });
  });
  chunks.push({
    id,
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: finish }],
    usage,
  });
  return (
    chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("") + "data: [DONE]\n\n"
  );
}

function anthropicContentBlocks(ans: NormalizedAnswer): Array<Record<string, unknown>> {
  const blocks: Array<Record<string, unknown>> = [];
  if (ans.text !== "") {
    blocks.push({ type: "text", text: ans.text });
  }
  for (const call of ans.toolCalls) {
    let input: Record<string, unknown> = {};
    try {
      const parsed: unknown = JSON.parse(call.args);
      if (isRecord(parsed)) input = parsed;
    } catch {
      input = {};
    }
    blocks.push({ type: "tool_use", id: call.id, name: call.name, input });
  }
  if (blocks.length === 0) {
    blocks.push({ type: "text", text: "" });
  }
  return blocks;
}

export function renderAnthropicAnswer(
  ans: NormalizedAnswer,
  model: string,
  id: string,
): Record<string, unknown> {
  const usage = repairedUsage(ans);
  return {
    id,
    type: "message",
    role: "assistant",
    content: anthropicContentBlocks(ans),
    model,
    stop_reason: anthropicStopReason(ans),
    stop_sequence: null,
    usage: {
      input_tokens: usage.prompt_tokens,
      output_tokens: usage.completion_tokens,
    },
  };
}

export function renderAnthropicStream(
  ans: NormalizedAnswer,
  model: string,
  id: string,
): string {
  const usage = repairedUsage(ans);
  const stop = anthropicStopReason(ans);
  const blocks = anthropicContentBlocks(ans);
  const events: Array<{ event: string; data: Record<string, unknown> }> = [];
  events.push({
    event: "message_start",
    data: {
      type: "message_start",
      message: {
        id,
        type: "message",
        role: "assistant",
        content: [],
        model,
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: usage.prompt_tokens,
          output_tokens: usage.completion_tokens,
        },
      },
    },
  });
  blocks.forEach((block, index) => {
    if (block.type === "text") {
      events.push({
        event: "content_block_start",
        data: {
          type: "content_block_start",
          index,
          content_block: { type: "text", text: "" },
        },
      });
      for (const piece of splitPieces(asText(block.text), 120)) {
        events.push({
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index,
            delta: { type: "text_delta", text: piece },
          },
        });
      }
      events.push({
        event: "content_block_stop",
        data: { type: "content_block_stop", index },
      });
    } else {
      const call = ans.toolCalls[index - (ans.text !== "" ? 1 : 0)];
      events.push({
        event: "content_block_start",
        data: {
          type: "content_block_start",
          index,
          content_block: {
            type: "tool_use",
            id: block.id,
            name: block.name,
            input: {},
          },
        },
      });
      events.push({
        event: "content_block_delta",
        data: {
          type: "content_block_delta",
          index,
          delta: { type: "input_json_delta", partial_json: call ? call.args : "{}" },
        },
      });
      events.push({
        event: "content_block_stop",
        data: { type: "content_block_stop", index },
      });
    }
  });
  events.push({
    event: "message_delta",
    data: {
      type: "message_delta",
      delta: { stop_reason: stop, stop_sequence: null },
      usage: {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
      },
    },
  });
  events.push({ event: "message_stop", data: { type: "message_stop" } });
  return events
    .map((e) => `event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`)
    .join("");
}

// GET /v1/models in the wire the auth header names. Both shapes list
// exactly the catalog ids, once each, in ladder order.
export function renderOpenAIModels(
  models: Array<{ id: string }>,
): Record<string, unknown> {
  return {
    object: "list",
    data: models.map((m) => ({
      id: m.id,
      object: "model",
      created: FIXED_CREATED,
      owned_by: "openrouter",
    })),
  };
}

export function renderAnthropicModels(
  models: Array<{ id: string }>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {
    data: models.map((m) => ({
      id: m.id,
      type: "model",
      display_name: m.id,
      created_at: FIXED_CREATED_AT,
    })),
    has_more: false,
  };
  if (models.length > 0) {
    out.first_id = models[0].id;
    out.last_id = models[models.length - 1].id;
  }
  return out;
}
