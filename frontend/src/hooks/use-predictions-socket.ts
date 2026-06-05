import {
  getPredictionStatusEvent,
  type PredictionStatusEvent as PredictionStatus,
} from "@/lib/backend-contract";
import { useWebSocket } from "./use-websocket";

export type { PredictionStatus };

export function usePredictionsSocket() {
  const socket = useWebSocket("/api/v1/ws/predictions");

  return {
    ...socket,
    events: socket.events.flatMap((event) => {
      const status = getPredictionStatusEvent(event);
      return status ? [status] : [];
    }),
    last: getPredictionStatusEvent(socket.last),
  };
}
