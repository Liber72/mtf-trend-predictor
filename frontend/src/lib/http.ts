import axios, { AxiosError } from "axios";

type EnvWithApi = ImportMetaEnv & {
  VITE_API_BASE_URL?: string;
  VITE_WS_BASE_URL?: string;
};

type WindowWithApi = Window & {
  __API_BASE__?: string;
};

type ValidationErrorItem = {
  msg?: string;
};

const baseURL =
  (typeof window !== "undefined" && (window as WindowWithApi).__API_BASE__) ||
  (import.meta.env as EnvWithApi).VITE_API_BASE_URL ||
  "http://localhost:8000";

export const http = axios.create({
  baseURL,
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

export function getErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = err.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          const message = (item as ValidationErrorItem).msg;
          return message || JSON.stringify(item);
        })
        .join("; ");
    }
    if (err.message) return err.message;
  }
  if (err instanceof Error) return err.message;
  return "Unknown error";
}

export function getWsUrl(path: string): string {
  const explicit = (import.meta.env as EnvWithApi).VITE_WS_BASE_URL;
  if (explicit) return explicit + path;
  try {
    const u = new URL(baseURL);
    const proto = u.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${u.host}${path}`;
  } catch {
    return `ws://localhost:8000${path}`;
  }
}

export const API_BASE = baseURL;
