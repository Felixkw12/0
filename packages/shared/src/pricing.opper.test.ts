import { describe, expect, it } from "vitest";
import { getRates, modelProvider } from "./pricing.js";

describe("Opper pricing and routing identity", () => {
  it("keeps opper/ model IDs on the Opper connection", () => {
    expect(modelProvider("opper/gemini-3.8-flash")).toBe("opper");
    expect(modelProvider("opper/anthropic/claude-sonnet-4-6")).toBe("opper");
  });
  it("prices an Opper route at the underlying model's rate instead of the generic default", () => {
    expect(getRates("opper/gemini-3.8-flash")).toEqual(getRates("gemini-3.8-flash"));
    expect(getRates("opper/anthropic/claude-sonnet-4-6")).toEqual(getRates("claude-sonnet-4-6"));
    expect(getRates("opper/gemini-3.8-flash")).not.toEqual(getRates());
  });
});
