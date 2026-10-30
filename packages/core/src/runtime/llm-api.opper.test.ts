import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LlmApiRuntime, parseLlmFallbackChain, resolveFailoverProvider } from "./llm-api.js";
import type { NativeMessage, NativeToolDef } from "./types.js";

const messages: NativeMessage[] = [{ role: "user", content: [{ type: "text", text: "Inspect the source" }] }];
const tools: NativeToolDef[] = [{ name: "read_file", description: "Read source", input_schema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } }];
const completion = { choices: [{ message: { content: "Opper answer" }, finish_reason: "stop" }], usage: { prompt_tokens: 12, completion_tokens: 4, prompt_tokens_details: { cached_tokens: 3 } } };
const env = { OPPER_API_KEY: "synthetic-opper-key", ZERO_LLM_FALLBACK: "", ZERO_SKIP_PROVIDER_BANNER: "1" };
function runtime(model?: string, extra: Record<string, string> = {}) {
  return new LlmApiRuntime({ type: "api", provider: "opper", ...(model ? { model } : {}), timeout: 1000, env: { ...env, ...extra } });
}

beforeEach(() => {
  for (const key of ["ZERO_SELECTED_PROVIDER", "ZERO_FORCE_PROVIDER", "ZERO_MODEL", "ZERO_LLM_FALLBACK", "ZERO_WORKBENCH_PROVIDER_PROXY", "ZERO_WORKBENCH_PROVIDER_PROXY_TOKEN", "OPPER_API_KEY", "OPPER_BASE_URL"]) vi.stubEnv(key, undefined);
});
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); vi.restoreAllMocks(); });

describe("Opper transport", () => {
  it.each(["ordinary", "native"])("defaults to the Opper Chat Completions endpoint on the %s path", async path => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(Response.json(completion));
    vi.stubGlobal("fetch", fetchMock);
    const rt = runtime();
    expect(rt.resolvedProvider()).toBe("opper");
    expect(rt.resolvedModel()).toBe("claude-sonnet-4-6");
    if (path === "ordinary") expect(await rt.execute("Review")).toMatchObject({ exitCode: 0, output: "Opper answer" });
    else expect(await rt.executeNative("Review", messages, [])).toMatchObject({ content: [{ type: "text", text: "Opper answer" }], stopReason: "end_turn", usage: { inputTokens: 12, outputTokens: 4, cachedInputTokens: 3 } });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://api.opper.ai/v3/compat/chat/completions");
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer synthetic-opper-key");
    expect(new Headers(init?.headers).has("x-api-key")).toBe(false);
    expect(JSON.parse(String(init?.body))).toMatchObject({ model: "claude-sonnet-4-6" });
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it.each([
    ["opper/claude-sonnet-4-6", "claude-sonnet-4-6"],
    ["opper/anthropic/claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"],
    ["gemini-3.8-flash", "gemini-3.8-flash"],
  ])("honors OPPER_BASE_URL and strips only the routing prefix from %s", async (model, upstream) => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(Response.json(completion));
    vi.stubGlobal("fetch", fetchMock);
    const rt = runtime(model, { OPPER_BASE_URL: "https://selected.example/v3/compat" });
    await rt.executeNative("Review", messages, []);
    expect(rt.resolvedModel()).toBe(upstream);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://selected.example/v3/compat/chat/completions");
    expect(JSON.parse(String(init?.body))).toMatchObject({ model: upstream, max_tokens: 8192 });
  });

  it("sends GPT-6 family IDs with max_completion_tokens on the Chat Completions wire", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(Response.json(completion));
    vi.stubGlobal("fetch", fetchMock);
    await runtime("gpt-6.1-terra").executeNative("Review", messages, tools);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://api.opper.ai/v3/compat/chat/completions");
    const body = JSON.parse(String(init?.body));
    expect(body).toMatchObject({ model: "gpt-6.1-terra", max_completion_tokens: 8192 });
    expect(body).not.toHaveProperty("max_tokens");
  });

  it("continues a conversation after a tool result on the same connection", async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(Response.json({
        choices: [{ message: { content: null, tool_calls: [{ id: "call-read", type: "function", function: { name: "read_file", arguments: '{"path":"src/main.ts"}' } }] }, finish_reason: "tool_calls" }],
      }))
      .mockResolvedValueOnce(Response.json(completion));
    vi.stubGlobal("fetch", fetchMock);
    const rt = runtime();
    const first = await rt.executeNative("Review", messages, tools);
    expect(first).toMatchObject({ stopReason: "tool_use", content: [{ type: "tool_use", id: "call-read", name: "read_file", input: { path: "src/main.ts" } }] });
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body)).tools[0].function).toMatchObject({ name: "read_file", parameters: { type: "object", required: ["path"] } });
    const result = await rt.executeNative("Review", [...messages,
      { role: "assistant", content: first.content },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "call-read", content: "export const answer = 42;" }] },
    ], tools);
    expect(result.stopReason).toBe("end_turn");
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({ model: "claude-sonnet-4-6",
      messages: expect.arrayContaining([
        { role: "assistant", content: null, tool_calls: [{ id: "call-read", type: "function", function: { name: "read_file", arguments: '{"path":"src/main.ts"}' } }] },
        { role: "tool", tool_call_id: "call-read", content: "export const answer = 42;" },
      ]),
    });
  });

  it("lists models from the connection's own /models endpoint", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(Response.json({ data: [
      { id: "claude-sonnet-4-6", context_length: 1_000_000, opper: { kind: "pool", type: "llm" } },
      { id: "anthropic/claude-sonnet-4-6", context_length: 1_000_000, opper: { kind: "model", type: "llm" } },
      { id: "text-embedding-3-large", opper: { kind: "pool", type: "embedding" } },
    ] }));
    vi.stubGlobal("fetch", fetchMock);
    expect(await runtime().availableModelCatalog()).toEqual([
      { id: "claude-sonnet-4-6", contextTokens: 1_000_000 },
      { id: "anthropic/claude-sonnet-4-6", contextTokens: 1_000_000 },
    ]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://api.opper.ai/v3/compat/models");
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer synthetic-opper-key");
  });

  it("labels configuration diagnostics as Opper", () => {
    expect(runtime().getConfigurationDiagnostics()).toMatchObject({ valid: true, provider: "opper", providerLabel: "Opper" });
  });
});

describe("Opper routing", () => {
  it("routes an explicit opper/ model ahead of other configured keys and leaves bare IDs on their natural route", () => {
    const keys = { OPPER_API_KEY: "synthetic-opper-key", ANTHROPIC_API_KEY: "synthetic-anthropic-key", ZERO_SKIP_PROVIDER_BANNER: "1", ZERO_LLM_FALLBACK: "" };
    const prefixed = new LlmApiRuntime({ type: "api", model: "opper/claude-sonnet-4-6", env: keys });
    expect(prefixed.resolvedProvider()).toBe("opper");
    expect(prefixed.resolvedModel()).toBe("claude-sonnet-4-6");
    const bare = new LlmApiRuntime({ type: "api", model: "claude-sonnet-4-6", env: keys });
    expect(bare.resolvedProvider()).toBe("anthropic");
  });

  it("fails closed for opper/ IDs without Opper auth even when other providers are configured", () => {
    const model = "opper/claude-sonnet-4-6";
    expect(() => new LlmApiRuntime({ type: "api", model, env: { OPPER_API_KEY: "", OPENAI_API_KEY: "synthetic-other-key", ZERO_SKIP_PROVIDER_BANNER: "1" } })).toThrow(/Opper model requires OPPER_API_KEY/);
    expect(() => new LlmApiRuntime({ type: "api", model, apiKey: "synthetic-ambiguous-key", env: { OPPER_API_KEY: "", ZERO_SKIP_PROVIDER_BANNER: "1" } })).toThrow(/explicit Opper provider/);
    expect(() => new LlmApiRuntime({ type: "api", provider: "openai", model, env: { OPPER_API_KEY: "synthetic-key", OPENAI_API_KEY: "synthetic-other-key", ZERO_SKIP_PROVIDER_BANNER: "1" } })).toThrow(/require the Opper provider/);
  });

  it("requires the Opper connection's own nonblank credential for failover", () => {
    expect(resolveFailoverProvider("opper", "claude-sonnet-4-6", { OPENAI_API_KEY: "synthetic-other-key" })).toBeUndefined();
    expect(resolveFailoverProvider("opper", "claude-sonnet-4-6", { OPPER_API_KEY: "   " })).toBeUndefined();
    expect(resolveFailoverProvider("opper", "gpt-6.1-terra", { OPPER_API_KEY: "synthetic-opper-key" })).toEqual({
      apiKey: "synthetic-opper-key", baseUrl: "https://api.opper.ai/v3/compat", wireApi: "chat_completions",
    });
    expect(parseLlmFallbackChain({ ZERO_LLM_FALLBACK: "opper:opper/gemini-3.8-flash" })).toEqual([
      { provider: "opper", model: "opper/gemini-3.8-flash" },
    ]);
  });

  it("strips the routing prefix when switching model and forking", async () => {
    const rt = runtime();
    rt.reconfigure({ model: "opper/gemini-3.8-flash" });
    expect(rt.resolvedModel()).toBe("gemini-3.8-flash");
    const child = await rt.forkForSubagent(500);
    expect(child.resolvedModel()).toBe("gemini-3.8-flash");
    expect(child.resolvedProvider()).toBe("opper");
  });
});
