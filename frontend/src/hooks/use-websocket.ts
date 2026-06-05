import { useEffect, useRef, useState } from "react";
import { getWsUrl } from "@/lib/http";

export type SocketStatus = "connecting" | "connected" | "reconnecting" | "disconnected";

export interface SocketState<T> {
  status: SocketStatus;
  events: T[];
  last: T | null;
}

const BACKOFF = [1000, 2000, 5000, 10_000];
const BUFFER = 200;

export function useWebSocket<T = unknown>(path: string): SocketState<T> {
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const [events, setEvents] = useState<T[]>([]);
  const [last, setLast] = useState<T | null>(null);
  const attemptRef = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | null>(null);
  const closedByUser = useRef(false);

  useEffect(() => {
    closedByUser.current = false;
    const connect = () => {
      try {
        setStatus(attemptRef.current === 0 ? "connecting" : "reconnecting");
        const ws = new WebSocket(getWsUrl(path));
        wsRef.current = ws;
        ws.onopen = () => {
          attemptRef.current = 0;
          setStatus("connected");
        };
        ws.onmessage = (ev) => {
          try {
            const data = JSON.parse(ev.data);
            setLast(data);
            setEvents((prev) => {
              const next = [...prev, data];
              return next.length > BUFFER ? next.slice(-BUFFER) : next;
            });
          } catch {
            // ignore non-json frames
          }
        };
        ws.onerror = () => {
          /* let onclose handle reconnect */
        };
        ws.onclose = () => {
          if (closedByUser.current) {
            setStatus("disconnected");
            return;
          }
          setStatus("reconnecting");
          const delay = BACKOFF[Math.min(attemptRef.current, BACKOFF.length - 1)];
          attemptRef.current += 1;
          timerRef.current = window.setTimeout(connect, delay);
        };
      } catch {
        setStatus("reconnecting");
        const delay = BACKOFF[Math.min(attemptRef.current, BACKOFF.length - 1)];
        attemptRef.current += 1;
        timerRef.current = window.setTimeout(connect, delay);
      }
    };
    connect();
    return () => {
      closedByUser.current = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
      wsRef.current?.close();
    };
  }, [path]);

  return { status, events, last };
}
