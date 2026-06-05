import { useWebSocket } from "./use-websocket";

export interface TradeEvent {
  type?: string;
  [k: string]: unknown;
}

export function useTradesSocket() {
  return useWebSocket<TradeEvent>("/api/v1/ws/trades");
}
