import { afterEach, describe, expect, test, vi } from "vitest";
import {
  highlight, parseOrderInput, searchWeeks, thoughtGroups,
} from "../src/app.js";
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

describe("thought tracker", () => {
  const weeks = Array.from({ length: 36 }, (_, index) => ({
    week: index + 1,
    quantity: index,
    thought: index === 13 ? "Backlog is finally clearing." : `Note ${index + 1}`,
  }));

  test("splits the episode into six-week chapters that cover every week", () => {
    const groups = thoughtGroups(weeks);
    expect(groups).toHaveLength(6);
    expect(groups.every((group) => group.length === 6)).toBe(true);
    expect(groups.flat().map((entry) => entry.week)).toEqual(weeks.map((entry) => entry.week));
  });

  test("search matches note text, a bare week number, or nothing at all", () => {
    expect(searchWeeks(weeks, "  ")).toBeNull();
    expect(searchWeeks(weeks, "backlog").map((entry) => entry.week)).toEqual([14]);
    expect(searchWeeks(weeks, "week 14").map((entry) => entry.week)).toEqual([14]);
    expect(searchWeeks(weeks, "capacity")).toEqual([]);
  });

  test("highlighting escapes the note and the query it injects", () => {
    expect(highlight("<b>ordered</b>", "")).toBe("&lt;b&gt;ordered&lt;/b&gt;");
    expect(highlight("hold & ship", "&")).toBe("hold <mark>&amp;</mark> ship");
    expect(highlight("a+b", "+")).toBe("a<mark>+</mark>b");
    expect(highlight("Don't panic", "39")).toBe("Don&#039;t panic");
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
