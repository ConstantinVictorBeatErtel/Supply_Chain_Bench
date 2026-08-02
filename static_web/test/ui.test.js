import { afterEach, describe, expect, test, vi } from "vitest";
import { parseOrderInput } from "../src/app.js";
import {
  configureTelemetry, createSessionUuid, sendTelemetry,
} from "../src/telemetry.js";

afterEach(() => {
  vi.unstubAllGlobals();
  configureTelemetry("");
});

describe("order input", () => {
  test("accepts only canonical integer text in the closed interval", () => {
    expect(parseOrderInput("0")).toEqual({ ok: true, quantity: 0 });
    expect(parseOrderInput("128")).toEqual({ ok: true, quantity: 128 });
    for (const invalid of ["", " ", "-1", "129", "1.0", "1e2", "+8", "08", "true", "NaN"]) {
      expect(parseOrderInput(invalid).ok).toBe(false);
    }
  });
});

describe("fail-soft telemetry", () => {
  test("a synchronous or asynchronous logging failure never throws", async () => {
    configureTelemetry("https://logger.invalid/session");
    vi.stubGlobal("fetch", vi.fn(() => {
      throw new Error("offline");
    }));
    expect(() => sendTelemetry({ status: "completed" })).not.toThrow();
    expect(sendTelemetry({ status: "completed" })).toBe(false);

    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline"))));
    expect(sendTelemetry({ status: "completed" })).toBe(true);
    await Promise.resolve();
  });

  test("session identifiers contain no personal information", () => {
    expect(createSessionUuid()).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
  });
});
