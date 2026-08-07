/**
 * macaron-websearch — custom provider extension
 *
 * Enables Anthropic-native server-side web_search on the macaron-anthropic
 * endpoint by:
 *   1. re-registering the `macaron-anthropic` provider (read from
 *      ~/.pi/agent/models.json) under a dedicated api name so the built-in
 *      `anthropic-messages` handler is left untouched;
 *   2. injecting the `web_search_20250305` server tool into every request via
 *      a custom `streamSimple`;
 *   3. parsing the `server_tool_use` and `web_search_tool_result` content
 *      blocks the endpoint streams back and rendering them as `text` blocks.
 *
 * Rendering server-tool blocks as text keeps pi's context model
 * (text | thinking | toolCall) intact and lets the assistant message round-
 * trip cleanly on the next turn — the model sees its own prior search results
 * as text, so we never have to echo raw server-tool blocks back to the API.
 *
 * Env knobs:
 *   PI_WEBSEARCH_DISABLE=1            -> skip this extension entirely
 *   PI_WEBSEARCH_TOOL_TYPE            -> default "web_search_20250305"
 *   PI_WEBSEARCH_MAX_USES             -> default 5
 *   PI_WEBSEARCH_ALLOWED_DOMAINS     -> comma-separated, optional
 *   PI_WEBSEARCH_BLOCKED_DOMAINS      -> comma-separated, optional
 *   PI_MACARON_PROVIDER              -> provider name to wrap (default macaron-anthropic)
 */

import Anthropic from "@anthropic-ai/sdk";
import {
	calculateCost,
	createAssistantMessageEventStream,
	parseStreamingJson,
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream as EventStream,
	type Context,
	type ImageContent,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type StopReason,
	type TextContent,
	type ThinkingContent,
	type Tool,
	type ToolCall,
	type ToolResultMessage,
} from "@mariozechner/pi-ai";
import { defineTool, type ExtensionAPI, type ProviderConfig } from "@mariozechner/pi-coding-agent";
import { Type } from "@mariozechner/pi-ai";
import TurndownService from "turndown";
import { Readability } from "@mozilla/readability";
import { JSDOM } from "jsdom";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// helpers (mirror the built-in anthropic provider so behavior stays faithful
// for the API-key, non-OAuth, non-Copilot path)
// ---------------------------------------------------------------------------

function sanitizeSurrogates(text: string): string {
	return text.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
}

function normalizeToolCallId(id: string): string {
	return id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
}

function convertContentBlocks(content: (TextContent | ImageContent)[]) {
	const hasImages = content.some((c) => c.type === "image");
	if (!hasImages) {
		return sanitizeSurrogates(content.map((c) => (c as TextContent).text).join("\n"));
	}
	const blocks = content.map((block) =>
		block.type === "text"
			? { type: "text" as const, text: sanitizeSurrogates(block.text) }
			: { type: "image" as const, source: { type: "base64" as const, media_type: block.mimeType, data: block.data } },
	);
	if (!blocks.some((b) => b.type === "text")) {
		blocks.unshift({ type: "text" as const, text: "(see attached image)" });
	}
	return blocks;
}

function convertMessages(messages: Message[]): any[] {
	const params: any[] = [];
	for (let i = 0; i < messages.length; i++) {
		const msg = messages[i];

		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				if (msg.content.trim()) params.push({ role: "user", content: sanitizeSurrogates(msg.content) });
			} else {
				const blocks = msg.content
					.map((item) =>
						item.type === "text"
							? { type: "text" as const, text: sanitizeSurrogates(item.text) }
							: { type: "image" as const, source: { type: "base64" as const, media_type: item.mimeType, data: item.data } },
					)
					.filter((b) => (b.type === "text" ? b.text.trim().length > 0 : true));
				if (blocks.length > 0) params.push({ role: "user", content: blocks });
			}
		} else if (msg.role === "assistant") {
			const blocks: any[] = [];
			for (const block of msg.content) {
				if (block.type === "text" && block.text.trim()) {
					blocks.push({ type: "text", text: sanitizeSurrogates(block.text) });
				} else if (block.type === "thinking") {
					if ((block as ThinkingContent).redacted) {
						blocks.push({ type: "redacted_thinking", data: (block as ThinkingContent).thinkingSignature });
					} else if (block.thinking.trim()) {
						if ((block as ThinkingContent).thinkingSignature) {
							blocks.push({ type: "thinking", thinking: sanitizeSurrogates(block.thinking), signature: (block as ThinkingContent).thinkingSignature });
						} else {
							blocks.push({ type: "text", text: sanitizeSurrogates(block.thinking) });
						}
					}
				} else if (block.type === "toolCall") {
					blocks.push({ type: "tool_use", id: normalizeToolCallId(block.id), name: block.name, input: block.arguments ?? {} });
				}
			}
			if (blocks.length > 0) params.push({ role: "assistant", content: blocks });
		} else if (msg.role === "toolResult") {
			const toolResults: any[] = [{
				type: "tool_result",
				tool_use_id: normalizeToolCallId((msg as ToolResultMessage).toolCallId),
				content: convertContentBlocks((msg as ToolResultMessage).content),
				is_error: (msg as ToolResultMessage).isError,
			}];
			let j = i + 1;
			while (j < messages.length && messages[j].role === "toolResult") {
				const n = messages[j] as ToolResultMessage;
				toolResults.push({ type: "tool_result", tool_use_id: normalizeToolCallId(n.toolCallId), content: convertContentBlocks(n.content), is_error: n.isError });
				j++;
			}
			i = j - 1;
			params.push({ role: "user", content: toolResults });
		}
	}

	// cache_control on last user message
	if (params.length > 0) {
		const last = params[params.length - 1];
		if (last.role === "user" && Array.isArray(last.content)) {
			const lb = last.content[last.content.length - 1];
			if (lb && (lb.type === "text" || lb.type === "image" || lb.type === "tool_result")) {
				lb.cache_control = { type: "ephemeral" };
			}
		}
	}
	return params;
}

function convertTools(tools: Tool[]): any[] {
	return tools.map((tool) => ({
		name: tool.name,
		description: tool.description,
		input_schema: { type: "object", properties: (tool.parameters as any).properties || {}, required: (tool.parameters as any).required || [] },
	}));
}

function mapStopReason(reason: string): StopReason {
	switch (reason) {
		case "end_turn":
		case "pause_turn":
		case "stop_sequence":
			return "stop";
		case "max_tokens":
			return "length";
		case "tool_use":
			return "toolUse";
		case "refusal":
		case "sensitive":
			return "error";
		default:
			throw new Error(`Unhandled stop reason: ${reason}`);
	}
}

// ---------------------------------------------------------------------------
// server-side tool injection config
// ---------------------------------------------------------------------------

function toolEnabled(envName: string, defaultValue = true): boolean {
	const raw = process.env[envName];
	if (raw === undefined) return defaultValue;
	const v = raw.trim().toLowerCase();
	return !(v === "0" || v === "false" || v === "no" || v === "off");
}

function buildWebSearchTool(): any {
	const tool: any = {
		type: process.env.PI_WEBSEARCH_TOOL_TYPE || "web_search_20250305",
		name: "web_search",
	};
	const maxUses = Number(process.env.PI_WEBSEARCH_MAX_USES ?? 5);
	if (Number.isFinite(maxUses) && maxUses > 0) tool.max_uses = maxUses;
	const allowed = process.env.PI_WEBSEARCH_ALLOWED_DOMAINS;
	if (allowed && allowed.trim()) tool.allowed_domains = allowed.split(",").map((d) => d.trim()).filter(Boolean);
	const blocked = process.env.PI_WEBSEARCH_BLOCKED_DOMAINS;
	if (blocked && blocked.trim()) tool.blocked_domains = blocked.split(",").map((d) => d.trim()).filter(Boolean);
	return tool;
}

function buildWebFetchTool(): any {
	return { type: process.env.PI_WEBFETCH_TOOL_TYPE || "web_fetch_20250305", name: "web_fetch" };
}

function buildCodeExecutionTool(): any {
	return { type: process.env.PI_CODEEXEC_TOOL_TYPE || "code_execution_20250522", name: "code_execution" };
}

function buildServerTools(): any[] {
	const tools: any[] = [];
	if (toolEnabled("PI_TOOL_WEB_SEARCH")) tools.push(buildWebSearchTool());
	// web_fetch / code_execution do NOT auto-execute on the macaron endpoint:
	// the model emits a plain tool_use and waits for a client result pi can't
	// provide, stalling the turn. Default them OFF; flip on only when the
	// endpoint truly supports server-side execution (then the result-block
	// parsing below already handles them).
	if (toolEnabled("PI_TOOL_WEB_FETCH", false)) tools.push(buildWebFetchTool());
	if (toolEnabled("PI_TOOL_CODE_EXECUTION", false)) tools.push(buildCodeExecutionTool());
	return tools;
}

// ---------------------------------------------------------------------------
// server-tool result block rendering
// ---------------------------------------------------------------------------

function maxResultChars(): number {
	const n = Number(process.env.PI_SERVER_TOOL_RESULT_MAX_CHARS ?? 8000);
	return Number.isFinite(n) && n > 0 ? n : 8000;
}

function cap(s: string): string {
	const max = maxResultChars();
	return s.length > max ? s.slice(0, max) + `\n…[truncated ${s.length - max} chars]` : s;
}

function formatServerUse(name: string, input: any): string {
	let detail = "";
	if (name === "web_search" && input && typeof input.query === "string" && input.query.trim()) {
		detail = `"${input.query}"`;
	} else if (input && typeof input === "object" && Object.keys(input).length) {
		try { detail = JSON.stringify(input); } catch { detail = String(input); }
	}
	const icon = name === "web_search" ? "🔎" : name === "web_fetch" ? "🌐" : name === "code_execution" ? "⚡" : "🛠";
	return `\n\n${icon} [${name}] ${detail}\n`;
}

function formatSearchResult(block: any): string {
	const results: any[] = Array.isArray(block?.content) ? block.content : [];
	if (results.length === 0) return `\n[web_search_tool_result] (empty)\n`;
	const lines: string[] = ["\n[web_search_tool_result]"];
	const shown = results.slice(0, 8);
	for (let i = 0; i < shown.length; i++) {
		const r = shown[i];
		const title = typeof r?.title === "string" ? r.title : "(untitled)";
		const url = typeof r?.url === "string" ? r.url : "";
		const page = typeof r?.page_number === "number" ? ` (p.${r.page_number})` : "";
		lines.push(`  ${i + 1}. ${title}${page}`);
		if (url) lines.push(`     ${url}`);
	}
	if (results.length > shown.length) lines.push(`  …and ${results.length - shown.length} more`);
	lines.push("");
	return lines.join("\n");
}

function formatWebFetchResult(block: any): string {
	const c = block?.content || {};
	const url = typeof c.url === "string" ? c.url : block?.tool_use_id || "";
	const doc = c.content;
	let body = "";
	if (doc && typeof doc === "object") {
		const src = doc.source;
		if (src?.type === "text" && typeof src?.data === "string") body = src.data;
		else if (src?.type === "base64") body = "[base64-encoded content]";
		else if (typeof src?.url === "string") body = `[content at ${src.url}]`;
		else { try { body = JSON.stringify(doc, null, 2); } catch { body = String(doc); } }
	} else if (typeof c.content === "string") {
		body = c.content;
	} else if (typeof c === "string") {
		body = c;
	}
	if (!body) { try { body = JSON.stringify(c, null, 2); } catch { body = String(c); } }
	return `\n🌐 [web_fetch] ${url}\n${body}\n`;
}

function formatBashResult(block: any): string {
	const c = block?.content || {};
	const parts = [`\n⚡ [bash_code_execution] exit=${c.exit_code ?? "?"}${c.interrupted ? " (interrupted)" : ""}${c.timeout ? " (timeout)" : ""}`];
	if (typeof c.stdout === "string" && c.stdout.length) parts.push("--- stdout ---", c.stdout);
	if (typeof c.stderr === "string" && c.stderr.length) parts.push("--- stderr ---", c.stderr);
	if (parts.length === 1) parts.push("(no output)");
	parts.push("");
	return parts.join("\n");
}

function formatPythonResult(block: any): string {
	const c = block?.content || {};
	const parts = [`\n🐍 [python_code_execution] exit=${c.exit_code ?? "?"}`];
	if (typeof c.stdout === "string" && c.stdout.length) parts.push("--- stdout ---", c.stdout);
	if (typeof c.stderr === "string" && c.stderr.length) parts.push("--- stderr ---", c.stderr);
	if (parts.length === 1) parts.push("(no output)");
	parts.push("");
	return parts.join("\n");
}

function formatTextEditorResult(block: any): string {
	const c = block?.content || {};
	if (c.file_type || typeof c.content === "string") {
		const start = c.start_line ?? 1;
		const range = c.total_lines != null ? ` lines ${start}-${start + (c.num_lines ?? 0)}/${c.total_lines}` : "";
		return `\n📝 [text_editor view] ${c.file_type || "text"}${range}\n${typeof c.content === "string" ? c.content : ""}\n`;
	}
	const parts = ["\n📝 [text_editor str_replace]"];
	if (c.old_start != null) parts.push(`old: lines ${c.old_start}-${c.old_start + (Array.isArray(c.old_lines) ? c.old_lines.length : 0)}`);
	if (c.new_start != null) parts.push(`new: lines ${c.new_start}-${c.new_start + (Array.isArray(c.new_lines) ? c.new_lines.length : 0)}`);
	parts.push("");
	return parts.join("\n");
}

/** Generic fallback: render any unrecognized server-tool result block as
 *  truncated JSON so future/new types are surfaced instead of dropped. */
function formatServerResult(cb: any): string {
	const t = cb?.type;
	let body: string;
	switch (t) {
		case "web_search_tool_result": body = formatSearchResult(cb); break;
		case "web_fetch_tool_result": body = formatWebFetchResult(cb); break;
		case "bash_code_execution_tool_result": body = formatBashResult(cb); break;
		case "python_code_execution_tool_result": body = formatPythonResult(cb); break;
		case "text_editor_code_execution_tool_result": body = formatTextEditorResult(cb); break;
		default:
			try { body = cb?.content != null ? JSON.stringify(cb.content, null, 2) : JSON.stringify(cb, null, 2); } catch { body = String(cb); }
			body = `\n[${t || "server_tool_result"}]\n${body}\n`;
	}
	return cap(body);
}

// ---------------------------------------------------------------------------
// streamSimple
// ---------------------------------------------------------------------------

type MyBlock = (TextContent | ThinkingContent | ToolCall) & {
	index: number;
	partialJson?: string;
	kind?: "text" | "thinking" | "toolCall" | "srvUse" | "srvResult";
	srvName?: string;
};

// Set from models.json at load time so the stream function knows whether the
// endpoint expects Authorization: Bearer (authHeader:true) or x-api-key.
let PROVIDER_AUTH_HEADER = false;

function streamMacaronWebSearch(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): EventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const apiKey = options?.apiKey ?? process.env.ANTHROPIC_AUTH_TOKEN ?? "";
			if (!apiKey) throw new Error(`No API key for provider: ${model.provider}`);

			const betaFeatures = ["fine-grained-tool-streaming-2025-05-14", "interleaved-thinking-2025-05-14"];
			// macaron (and any provider with authHeader:true) expects
			// Authorization: Bearer, not x-api-key. The Anthropic SDK uses
			// x-api-key for apiKey and Bearer for authToken.
			const useBearer = PROVIDER_AUTH_HEADER === true;
			const client = new Anthropic({
				apiKey: useBearer ? null : apiKey,
				authToken: useBearer ? apiKey : undefined,
				baseURL: model.baseUrl,
				dangerouslyAllowBrowser: true,
				defaultHeaders: {
					accept: "application/json",
					"anthropic-dangerous-direct-browser-access": "true",
					"anthropic-beta": betaFeatures.join(","),
					...(model.headers || {}),
					...(options?.headers || {}),
				},
			});

			// build params (faithful to built-in buildParams, api-key path)
			const params: any = {
				model: model.id,
				messages: convertMessages(context.messages),
				max_tokens: options?.maxTokens || Math.min(model.maxTokens || 32000, 32000),
				stream: true,
			};
			if (context.systemPrompt) {
				params.system = [{ type: "text", text: sanitizeSurrogates(context.systemPrompt), cache_control: { type: "ephemeral" } }];
			}
			if (options?.temperature !== undefined && !(options?.reasoning && model.reasoning)) {
				params.temperature = options.temperature;
			}
			const toolsArr: any[] = context.tools ? convertTools(context.tools) : [];
			// inject Anthropic-native server-side tools (web_search / web_fetch / code_execution)
			for (const t of buildServerTools()) toolsArr.push(t);
			params.tools = toolsArr;

			if (model.reasoning) {
				if (options?.reasoning) {
					const budgets: Record<string, number> = { minimal: 1024, low: 2048, medium: 8192, high: 16384 };
					const lvl = options.reasoning === "xhigh" ? "high" : options.reasoning;
					const budget = options.thinkingBudgets?.[lvl as keyof typeof options.thinkingBudgets] ?? budgets[lvl] ?? 10240;
					params.thinking = { type: "enabled", budget_tokens: budget };
				} else {
					params.thinking = { type: "disabled" };
				}
			}
			if (options?.metadata?.user_id) params.metadata = { user_id: options.metadata.user_id };

			// honor onPayload if a higher layer wants to inspect/replace
			const nextParams = await (options as any)?.onPayload?.(params, model);
			const finalParams = nextParams !== undefined ? nextParams : params;

			const anthropicStream = client.messages.stream({ ...finalParams, stream: true }, { signal: options?.signal });
			stream.push({ type: "start", partial: output });

			const blocks = output.content as any[];

			for await (const event of anthropicStream) {
				if (event.type === "message_start") {
					const u = (event as any).message?.usage || {};
					output.usage.input = u.input_tokens || 0;
					output.usage.output = u.output_tokens || 0;
					output.usage.cacheRead = u.cache_read_input_tokens || 0;
					output.usage.cacheWrite = u.cache_creation_input_tokens || 0;
					output.usage.totalTokens = output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
					calculateCost(model, output.usage);
				} else if (event.type === "content_block_start") {
					const cb = (event as any).content_block;
					const idx = (event as any).index;
					if (cb.type === "text") {
						blocks.push({ type: "text", text: "", index: idx, kind: "text" });
						stream.push({ type: "text_start", contentIndex: blocks.length - 1, partial: output });
					} else if (cb.type === "thinking") {
						blocks.push({ type: "thinking", thinking: "", thinkingSignature: "", index: idx, kind: "thinking" });
						stream.push({ type: "thinking_start", contentIndex: blocks.length - 1, partial: output });
					} else if (cb.type === "redacted_thinking") {
						blocks.push({ type: "thinking", thinking: "[Reasoning redacted]", thinkingSignature: cb.data, redacted: true, index: idx, kind: "thinking" });
						stream.push({ type: "thinking_start", contentIndex: blocks.length - 1, partial: output });
					} else if (cb.type === "tool_use") {
						blocks.push({ type: "toolCall", id: cb.id, name: cb.name, arguments: cb.input ?? {}, partialJson: "", index: idx, kind: "toolCall" });
						stream.push({ type: "toolcall_start", contentIndex: blocks.length - 1, partial: output });
					} else if (cb.type === "server_tool_use") {
						// render as text so pi can store/echo it
						blocks.push({ type: "text", text: "", index: idx, kind: "srvUse", partialJson: cb.input ? JSON.stringify(cb.input) : "", srvName: cb.name || "server_tool" });
						stream.push({ type: "text_start", contentIndex: blocks.length - 1, partial: output });
					} else {
						// Any other server-side result block (web_search_tool_result,
						// web_fetch_tool_result, *_code_execution_tool_result, and
						// future types) — formatted by formatServerResult, with a
						// generic JSON fallback so nothing is silently dropped.
						const text = formatServerResult(cb);
						blocks.push({ type: "text", text, index: idx, kind: "srvResult" });
						stream.push({ type: "text_start", contentIndex: blocks.length - 1, partial: output });
						stream.push({ type: "text_delta", contentIndex: blocks.length - 1, delta: text, partial: output });
					}
				} else if (event.type === "content_block_delta") {
					const idx = blocks.findIndex((b) => b.index === (event as any).index);
					const block = blocks[idx];
					if (!block) continue;
					const delta = (event as any).delta;
					if (delta.type === "text_delta" && block.type === "text" && block.kind === "text") {
						block.text += delta.text;
						stream.push({ type: "text_delta", contentIndex: idx, delta: delta.text, partial: output });
					} else if (delta.type === "thinking_delta" && block.type === "thinking") {
						block.thinking += delta.thinking;
						stream.push({ type: "thinking_delta", contentIndex: idx, delta: delta.thinking, partial: output });
					} else if (delta.type === "input_json_delta") {
						if (block.kind === "toolCall") {
							block.partialJson = (block.partialJson || "") + delta.partial_json;
							block.arguments = parseStreamingJson(block.partialJson);
							stream.push({ type: "toolcall_delta", contentIndex: idx, delta: delta.partial_json, partial: output });
						} else if (block.kind === "srvUse") {
							block.partialJson = (block.partialJson || "") + delta.partial_json;
						}
					} else if (delta.type === "signature_delta" && block.type === "thinking") {
						block.thinkingSignature = (block.thinkingSignature || "") + delta.signature;
					}
				} else if (event.type === "content_block_stop") {
					const idx = blocks.findIndex((b) => b.index === (event as any).index);
					const block = blocks[idx];
					if (!block) continue;
					delete block.index;
					if (block.kind === "text" || block.kind === "srvResult") {
						stream.push({ type: "text_end", contentIndex: idx, content: block.text, partial: output });
					} else if (block.kind === "thinking") {
						stream.push({ type: "thinking_end", contentIndex: idx, content: block.thinking, partial: output });
					} else if (block.kind === "toolCall") {
						block.arguments = parseStreamingJson(block.partialJson || "");
						delete block.partialJson;
						stream.push({ type: "toolcall_end", contentIndex: idx, toolCall: block as ToolCall, partial: output });
					} else if (block.kind === "srvUse") {
						let input: any = undefined;
						try { input = block.partialJson ? JSON.parse(block.partialJson) : undefined; } catch {}
						delete block.partialJson;
						block.text = formatServerUse(block.srvName || "server_tool", input);
						delete block.srvName;
						block.kind = "text";
						stream.push({ type: "text_delta", contentIndex: idx, delta: block.text, partial: output });
						stream.push({ type: "text_end", contentIndex: idx, content: block.text, partial: output });
					}
				} else if (event.type === "message_delta") {
					const d = (event as any).delta;
					const u = (event as any).usage || {};
					if (d.stop_reason) output.stopReason = mapStopReason(d.stop_reason);
					if (u.input_tokens != null) output.usage.input = u.input_tokens;
					if (u.output_tokens != null) output.usage.output = u.output_tokens;
					if (u.cache_read_input_tokens != null) output.usage.cacheRead = u.cache_read_input_tokens;
					if (u.cache_creation_input_tokens != null) output.usage.cacheWrite = u.cache_creation_input_tokens;
					output.usage.totalTokens = output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
					calculateCost(model, output.usage);
				}
			}

			if (options?.signal?.aborted) throw new Error("Request was aborted");
			if (output.stopReason === "aborted" || output.stopReason === "error") throw new Error("An unknown error occurred");

			stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse", message: output });
			stream.end();
		} catch (error) {
			for (const block of output.content as MyBlock[]) delete block.index;
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : String(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
}

// ---------------------------------------------------------------------------
// client-side web_fetch tool
//
// The macaron endpoint does NOT auto-execute web_fetch server-side: when the
// web_fetch tool is present the model emits a plain tool_use (name=web_fetch)
// and stops, waiting for the client to return the page content. With no
// executor that stalls the turn. This client-side tool performs the fetch
// locally (HTTP + naive HTML->text) and returns the body, so the agent can
// actually read URLs. (On a real Anthropic endpoint that supports server-side
// web_fetch, disable this and inject the server tool via PI_TOOL_WEB_FETCH=1
// instead — the formatWebFetchResult parser already covers that path.)
// ---------------------------------------------------------------------------

// Strip noise by default: drop images and turn links into plain text (drop the
// URL, keep the anchor text). Saves tokens and avoids link-spam clutter.
// Per-call `keep_images` / `keep_links` params (and PI_WEBFETCH_KEEP_IMAGES /
// PI_WEBFETCH_KEEP_LINKS env as defaults) toggle each independently.
function envFlag(name: string, def = false): boolean {
	const v = (process.env[name] ?? (def ? "1" : "0")).trim().toLowerCase();
	return v === "1" || v === "true" || v === "yes" || v === "on";
}

function makeTurndown(keepImages: boolean, keepLinks: boolean): TurndownService {
	const td = new TurndownService({ headingStyle: "atx", codeBlockStyle: "fenced", bulletListMarker: "-", emDelimiter: "*" });
	td.remove(["script", "style", "noscript", "iframe", "svg", "canvas", "template"]);
	if (!keepImages) {
		td.addRule("stripImages", { filter: "img", replacement: () => "" });
		td.addRule("stripPicture", { filter: "picture", replacement: () => "" });
	}
	if (!keepLinks) {
		td.addRule("stripLinkUrls", {
			filter: (node) => node.nodeName === "A" && node.getAttribute("href") != null,
			replacement: (content) => content,
		});
	}
	return td;
}

function readabilityEnabled(): boolean {
	const v = (process.env.PI_WEBFETCH_READABILITY ?? "1").trim().toLowerCase();
	return !(v === "0" || v === "false" || v === "no" || v === "off");
}

// Convert HTML to Markdown. Tries Readability first (strips nav/sidebars/
// ads/footers, keeps article body), then falls back to converting the whole
// page with Turndown. Readability is skipped for non-article pages (lists,
// dashboards) where it would drop too much — a too-short extraction triggers
// the fallback automatically.
function htmlToMarkdown(html: string, url: string, keepImages: boolean, keepLinks: boolean): string {
	const td = makeTurndown(keepImages, keepLinks);
	if (readabilityEnabled()) {
		try {
			const doc = new JSDOM(html, { url }).window.document;
			const article = new Readability(doc, { charThreshold: 200 }).parse();
			if (article?.content && article.content.replace(/<[^>]+>/g, "").trim().length > 200) {
				return td.turndown(article.content).trim();
			}
		} catch {
			// fall through to full-page turndown
		}
	}
	try {
		return td.turndown(html).trim();
	} catch {
		// last resort: crude tag-strip
		return html.replace(/<script[\s\S]*?<\/script>/gi, " ").replace(/<style[\s\S]*?<\/style>/gi, " ").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
	}
}

async function fetchUrl(url: string, maxLength: number, keepImages: boolean, keepLinks: boolean): Promise<string> {
	const res = await fetch(url, {
		redirect: "follow",
		headers: {
			"user-agent": "Mozilla/5.0 (compatible; pi-webfetch/1.0)",
			accept: "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.5",
		},
	});
	if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
	const ctype = (res.headers.get("content-type") || "").toLowerCase();
	const raw = await res.text();
	let body: string;
	if (ctype.includes("application/json") || ctype.includes("text/plain")) {
		body = raw;
	} else {
		body = htmlToMarkdown(raw, url, keepImages, keepLinks);
	}
	body = body.trim();
	if (body.length > maxLength) body = body.slice(0, maxLength) + `\n…[truncated ${body.length - maxLength} chars]`;
	return body;
}

/** Grep the fetched body: keep only matching lines. Pattern is treated as
 *  a (grep -E style) regex; if it fails to compile, falls back to literal
 *  substring match. No binary dependency. Returns matching lines joined by
 *  newlines, or empty when nothing matches (never throws on no-match). */
function grepFilter(input: string, pattern: string): string {
	if (!pattern.trim()) return input;
	let re: RegExp;
	try { re = new RegExp(pattern, "i"); } catch { re = new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"); }
	const lines = input.split("\n");
	const matched = lines.filter((line) => re.test(line));
	return matched.join("\n");
}

const clientWebFetchTool = defineTool({
	name: "web_fetch",
	label: "Web Fetch (client-side)",
	description:
		"Fetch a single URL and return its text content (HTML is converted to clean Markdown via Readability+Turndown). By default images and link URLs are stripped (anchor text kept) to save tokens — set keep_images=true and/or keep_links=true to preserve them. Optional grep pattern keeps only matching lines of the fetched body (piped via stdin). Returns the (grep-filtered) body text, truncated to max_length.",
	promptSnippet: "web_fetch(url, grep?, keep_images?, keep_links?): fetch a URL, optionally grep-filter and control media/link stripping.",
	parameters: Type.Object({
		url: Type.String({ description: "The absolute URL to fetch (http/https)." }),
		max_length: Type.Optional(Type.Number({ description: "Max chars to return (default 8000)." })),
		grep: Type.Optional(Type.String({ description: "Optional grep pattern. Only matching lines of the fetched body are kept (body piped via stdin)." })),
		keep_images: Type.Optional(Type.Boolean({ description: "Keep images in output (default false = strip). Override PI_WEBFETCH_KEEP_IMAGES env." })),
		keep_links: Type.Optional(Type.Boolean({ description: "Keep link URLs in output as [text](url) (default false = strip URLs, keep anchor text). Override PI_WEBFETCH_KEEP_LINKS env." })),
	}),
	async execute(_toolCallId, params, _signal) {
		try {
			const max = params.max_length ?? Number(process.env.PI_WEBFETCH_MAX_CHARS ?? 8000);
			const keepImages = params.keep_images ?? envFlag("PI_WEBFETCH_KEEP_IMAGES", false);
			const keepLinks = params.keep_links ?? envFlag("PI_WEBFETCH_KEEP_LINKS", false);
			let body = await fetchUrl(params.url, max, keepImages, keepLinks);
			if (params.grep) {
				body = grepFilter(body, params.grep);
			}
			return { content: [{ type: "text" as const, text: body }], details: { url: params.url, length: body.length, keep_images: keepImages, keep_links: keepLinks, grep: params.grep } as any };
		} catch (e: any) {
			return { content: [{ type: "text" as const, text: `web_fetch failed: ${e?.message || String(e)}` }], details: { url: params.url, error: e?.message } as any, isError: true };
		}
	},
});

// ---------------------------------------------------------------------------
// entry point: re-register macaron-anthropic under a dedicated api name
// ---------------------------------------------------------------------------

function readModelsJson(): any {
	try {
		const p = join(homedir(), ".pi", "agent", "models.json");
		return JSON.parse(readFileSync(p, "utf8"));
	} catch {
		return null;
	}
}

export default function (pi: ExtensionAPI) {
	if (process.env.PI_WEBSEARCH_DISABLE === "1" || process.env.PI_WEBSEARCH_DISABLE === "true") {
		return; // leave the built-in anthropic-messages handler in place
	}

	const providerName = process.env.PI_MACARON_PROVIDER || "macaron-anthropic";
	const modelsJson = readModelsJson();
	const existing = modelsJson?.providers?.[providerName];
	if (!existing) {
		// nothing to wrap; fall back silently
		return;
	}

	// Re-declare the provider with a dedicated api so the built-in
	// anthropic-messages handler stays untouched. models.json remains the
	// source of truth for baseUrl / apiKey / model list.
	PROVIDER_AUTH_HEADER = existing.authHeader === true;
	const config: ProviderConfig = {
		baseUrl: existing.baseUrl,
		apiKey: existing.apiKey,
		authHeader: existing.authHeader ?? false,
		api: "anthropic-websearch",
		headers: existing.headers,
		streamSimple: streamMacaronWebSearch,
		models: (existing.models || []).map((m: any) => ({
			id: m.id,
			name: m.name,
			input: m.input || ["text", "image"],
			reasoning: m.reasoning ?? false,
			cost: m.cost || { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: m.contextWindow || 200000,
			maxTokens: m.maxTokens || 32000,
		})),
	};

	pi.registerProvider(providerName, config as any);

	// Client-side web_fetch: the endpoint doesn't auto-execute it server-side,
	// so we provide a real fetch executor. Default on; disable with
	// PI_WEBFETCH_CLIENT=0. (Don't enable PI_TOOL_WEB_FETCH at the same time,
	// which injects the server tool — names would clash.)
	if (toolEnabled("PI_WEBFETCH_CLIENT", true)) {
		pi.registerTool(clientWebFetchTool);
	}
}
