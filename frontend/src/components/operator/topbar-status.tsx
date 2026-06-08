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
    <div className="flex items-center gap-5 bg-white/[0.02] border border-white/5 px-5 py-2 rounded-full shadow-[inset_0_1px_1px_rgba(255,255,255,0.02)] backdrop-blur-3xl">
      {/* Backend Status */}
      <div className="flex items-center gap-2">
        <div className={`h-2 w-2 rounded-full ${backendTone === "success" ? "bg-emerald-500" : "bg-red-500"}`} />
        <span className="text-[12px] font-medium text-slate-300">Core</span>
      </div>
      
      <div className="w-[1px] h-3 bg-white/10" />

      {/* MT5 Live Ping */}
      <div className="flex items-center gap-2">
        <div className="relative flex h-2 w-2">
          {mt5.data?.connected && (
            <span className="animate-[ping_2s_ease-out_infinite] absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          )}
          <span className={`relative inline-flex rounded-full h-2 w-2 ${mt5.data?.connected ? "bg-emerald-500 shadow-[0_0_12px_#10b981]" : "bg-red-500"}`}></span>
        </div>
        <span className="text-[12px] font-medium text-slate-300">MT5 Live</span>
      </div>

      <div className="w-[1px] h-3 bg-white/10" />

      {/* Auto Trade Engine */}
      <div className="flex items-center gap-2">
        <div className={`h-2 w-2 rounded-full ${autoTone === "success" ? "bg-yellow-500 shadow-[0_0_12px_#eab308]" : "bg-slate-600"}`} />
        <span className="text-[12px] font-medium text-slate-300">Engine</span>
      </div>
    </div>
  );
}
