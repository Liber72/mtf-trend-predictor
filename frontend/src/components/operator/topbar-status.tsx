import { useQuery } from "@tanstack/react-query";
import { http } from "@/lib/http";
import { StatusBadge, type StatusTone } from "./status-badge";

export function TopbarStatus() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: async () => (await http.get("/api/v1/health")).data,
    refetchInterval: 15_000,
    retry: 0,
  });
  const mt5 = useQuery({
    queryKey: ["mt5-status"],
    queryFn: async () => (await http.get("/api/v1/mt5/status")).data,
    refetchInterval: 30_000,
    retry: 0,
  });
  const auto = useQuery({
    queryKey: ["auto-trade-status"],
    queryFn: async () => (await http.get("/api/v1/trading/auto/status")).data,
    refetchInterval: 30_000,
    retry: 0,
  });

  const backendTone: StatusTone = health.isError
    ? "danger"
    : health.data?.status === "ok"
      ? "success"
      : "warning";
  const mt5Tone: StatusTone = mt5.isError ? "danger" : mt5.data?.connected ? "success" : "neutral";
  const autoTone: StatusTone = auto.isError ? "danger" : auto.data?.running ? "success" : "neutral";

  return (
    <div className="flex items-center gap-2 overflow-x-auto">
      <StatusBadge tone={backendTone}>
        Backend {health.isLoading ? "…" : backendTone === "success" ? "OK" : "Down"}
      </StatusBadge>
      <StatusBadge tone={mt5Tone}>MT5 {mt5.data?.connected ? "Connected" : "Offline"}</StatusBadge>
      <StatusBadge tone={autoTone}>Auto {auto.data?.running ? "Running" : "Stopped"}</StatusBadge>
    </div>
  );
}
