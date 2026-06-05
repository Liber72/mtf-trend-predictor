export interface PredictionStatusEvent {
  h1_model_loaded: boolean;
  m5_model_loaded: boolean;
  current_mode: string | null;
}

type PredictionTimeframeKey = "h1" | "m5";

type RecordLike = Record<string, unknown>;

function isRecord(value: unknown): value is RecordLike {
  return typeof value === "object" && value !== null;
}

export function getPredictionStatusEvent(value: unknown): PredictionStatusEvent | null {
  if (!isRecord(value)) return null;

  const payload = isRecord(value.data) ? value.data : value;
  if (!isRecord(payload)) return null;

  return {
    h1_model_loaded: payload.h1_model_loaded === true,
    m5_model_loaded: payload.m5_model_loaded === true,
    current_mode: typeof payload.current_mode === "string" ? payload.current_mode : null,
  };
}

export function getPredictionTimeframeDirection(
  value: unknown,
  timeframe: PredictionTimeframeKey,
): string | null {
  if (!isRecord(value) || !isRecord(value[timeframe])) return null;
  const direction = value[timeframe].direction;
  return typeof direction === "string" ? direction : null;
}

export function getPredictionProbability(
  value: unknown,
  timeframe: PredictionTimeframeKey,
): number | null {
  if (!isRecord(value) || !isRecord(value[timeframe])) return null;
  const probability = value[timeframe].probability;
  return typeof probability === "number" ? probability : null;
}

export function getCombinedPrediction(value: unknown): {
  signal: string | null;
  confidence: number | null;
  reason: string | null;
} {
  if (!isRecord(value) || !isRecord(value.combined)) {
    return { signal: null, confidence: null, reason: null };
  }

  const combined = value.combined;
  return {
    signal: typeof combined.signal === "string" ? combined.signal : null,
    confidence: typeof combined.confidence === "number" ? combined.confidence : null,
    reason: typeof combined.reason === "string" ? combined.reason : null,
  };
}

export function getModelArtifactPath(value: unknown): string | null {
  if (!isRecord(value)) return null;
  if (typeof value.artifact_path === "string") return value.artifact_path;
  if (typeof value.model_path === "string") return value.model_path;
  return null;
}

export function getTradeOpenedAt(value: unknown): string | null {
  if (!isRecord(value)) return null;
  if (typeof value.entry_time === "string") return value.entry_time;
  if (typeof value.opened_at === "string") return value.opened_at;
  return null;
}

export function getTradePnl(value: unknown): number | null {
  if (!isRecord(value)) return null;
  if (typeof value.pnl === "number") return value.pnl;
  if (typeof value.profit === "number") return value.profit;
  return null;
}
