# pi-macaron-harness

A [Pi](https://github.com/badlogic/pi-mono) custom provider extension that enables
**Anthropic-native server-side `web_search`** (and full pass-through of `web_fetch` /
`code_execution` result blocks) on a `macaron-anthropic`-style endpoint.

Stock Pi cannot turn on an endpoint's native `web_search_20250305` server tool via
`models.json` alone — the built-in `anthropic-messages` handler also silently drops
the `server_tool_use` / `web_search_tool_result` blocks. This extension:

1. Reads `~/.pi/agent/models.json`, re-registers the `macaron-anthropic` provider under
   a dedicated api name (`anthropic-websearch`) with a custom `streamSimple`, so the
   built-in `anthropic-messages` handler stays untouched.
2. Injects the `web_search_20250305` server tool into every request.
3. Parses every `server_tool_use` and `*_tool_result` block the endpoint streams back
   (`web_search_tool_result`, `web_fetch_tool_result`,
   `bash_code_execution_tool_result`, `text_editor_code_execution_tool_result`,
   `python_code_execution_tool_result`, plus a generic JSON fallback for future types)
   and renders them as `text` blocks.

Rendering server-tool blocks as `text` keeps Pi's context model
(`text` | `thinking` | `toolCall`) intact and lets the assistant message round-trip
cleanly on the next turn — the model sees its own prior tool results as text, so we
never have to echo raw server-tool blocks back to the API.

## Install

```bash
pi install git:github.com/trotsky1997/pi-macaron-harness
```

> Requires a `macaron-anthropic` provider entry in `~/.pi/agent/models.json`
> (the extension reads `baseUrl` / `apiKey` / `authHeader` / model list from it and
> sets them as the default). Restart Pi after installing.

## Endpoint capability note

Tested against the macaron endpoint: only `web_search` auto-executes server-side
(returns `server_tool_use` + `web_search_tool_result`). `web_fetch` and
`code_execution` do **not** auto-execute there (the model emits a plain `tool_use`
and waits for a client result Pi can't provide, stalling the turn), so they are
injected **off by default**. The result-block parsing already covers them — flip on
when your endpoint truly supports server-side execution.

## Env knobs

| Var | Default | Purpose |
|-----|---------|---------|
| `PI_WEBSEARCH_DISABLE` | off | `=1` skip the whole extension |
| `PI_TOOL_WEB_SEARCH` | `1` (on) | inject `web_search` |
| `PI_TOOL_WEB_FETCH` | `0` (off) | inject `web_fetch` (enable if endpoint supports it) |
| `PI_TOOL_CODE_EXECUTION` | `0` (off) | inject `code_execution` (enable if endpoint supports it) |
| `PI_WEBSEARCH_TOOL_TYPE` | `web_search_20250305` | tool type override |
| `PI_WEBSEARCH_MAX_USES` | `5` | max searches per turn |
| `PI_WEBSEARCH_ALLOWED_DOMAINS` | — | comma-separated allow-list |
| `PI_WEBSEARCH_BLOCKED_DOMAINS` | — | comma-separated block-list |
| `PI_WEBFETCH_TOOL_TYPE` | `web_fetch_20250305` | tool type override |
| `PI_CODEEXEC_TOOL_TYPE` | `code_execution_20250522` | tool type override |
| `PI_SERVER_TOOL_RESULT_MAX_CHARS` | `8000` | per-result-block truncation |
| `PI_MACARON_PROVIDER` | `macaron-anthropic` | provider name to wrap |
