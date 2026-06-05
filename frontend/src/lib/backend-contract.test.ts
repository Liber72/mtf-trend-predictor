import { describe, expect, test } from "bun:test";

import {
  getCombinedPrediction,
  getModelArtifactPath,
  getPredictionProbability,
  getPredictionStatusEvent,
  getPredictionTimeframeDirection,
  getTradeOpenedAt,
  getTradePnl,
} from "./backend-contract";

describe("getPredictionStatusEvent", () => {
  test("reads nested websocket status payloads", () => {
    expect(
      getPredictionStatusEvent({
        type: "status",
        data: {
          h1_model_loaded: true,
          m5_model_loaded: false,
          current_mode: "dual",
        },
      }),
    ).toEqual({
      h1_model_loaded: true,
      m5_model_loaded: false,
      current_mode: "dual",
    });
  });

  test("reads flat status payloads", () => {
    expect(
      getPredictionStatusEvent({
        h1_model_loaded: false,
        m5_model_loaded: true,
        current_mode: "single_m5",
      }),
    ).toEqual({
      h1_model_loaded: false,
      m5_model_loaded: true,
      current_mode: "single_m5",
    });
  });
});

describe("prediction helpers", () => {
  const response = {
    h1: { direction: "UP", probability: 0.82 },
    m5: { direction: "DOWN", probability: 0.31 },
    combined: { signal: "WAIT", confidence: 0.61, reason: "mixed" },
  };

  test("reads timeframe directions and probabilities from nested response", () => {
    expect(getPredictionTimeframeDirection(response, "h1")).toBe("UP");
    expect(getPredictionProbability(response, "h1")).toBe(0.82);
    expect(getPredictionTimeframeDirection(response, "m5")).toBe("DOWN");
    expect(getPredictionProbability(response, "m5")).toBe(0.31);
  });

  test("reads combined prediction block", () => {
    expect(getCombinedPrediction(response)).toEqual({
      signal: "WAIT",
      confidence: 0.61,
      reason: "mixed",
    });
  });
});

describe("model and trade helpers", () => {
  test("prefers artifact_path and falls back to model_path", () => {
    expect(
      getModelArtifactPath({ artifact_path: "models/h1.keras", model_path: "legacy.keras" }),
    ).toBe("models/h1.keras");
    expect(getModelArtifactPath({ model_path: "legacy.keras" })).toBe("legacy.keras");
  });

  test("reads trade timestamps and pnl from backend field names", () => {
    expect(getTradeOpenedAt({ entry_time: "2026-06-05T10:00:00Z", opened_at: "legacy" })).toBe(
      "2026-06-05T10:00:00Z",
    );
    expect(getTradeOpenedAt({ opened_at: "legacy" })).toBe("legacy");
    expect(getTradePnl({ pnl: 12.5, profit: 9 })).toBe(12.5);
    expect(getTradePnl({ profit: -3.25 })).toBe(-3.25);
  });
});
