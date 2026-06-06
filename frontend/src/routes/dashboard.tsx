import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Play, Power, PlugZap, Activity, Cpu, Gauge } from "lucide-react";
import { toast } from "sonner";
import { http, getErrorMessage } from "@/lib/http";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/operator/status-badge";
import { PageHeader, PageSection } from "@/components/operator/page-section";
import { ConfirmAction } from "@/components/operator/confirm-action";
import { usePredictionsSocket } from "@/hooks/use-predictions-socket";

export const Route = createFileRoute("/dashboard")({
  head: () => ({ meta: [{ title: "Dashboard · Operator" }] }),
  component: DashboardPage,
});

function DashboardPage() {
  const qc = useQueryClient();
  const invalidateAll = () => {
    qc.invalidateQueries({ queryKey: ["health"] });
    qc.invalidateQueries({ queryKey: ["mt5-status"] });
    qc.invalidateQueries({ queryKey: ["auto-trade-status"] });
  };

  const health = useQuery({
    queryKey: ["health"],
    queryFn: async () => (await http.get("/api/v1/health")).data,
    refetchInterval: 15_000,
    retry: 0,
  });
  const mt5 = useQuery({
    queryKey: ["mt5-status"],
    queryFn: async () => (await http.get("/api/v1/mt5/status")).data,
    retry: 0,
  });
  const auto = useQuery({
    queryKey: ["auto-trade-status"],
    queryFn: async () => (await http.get("/api/v1/trading/auto/status")).data,
    retry: 0,
  });
  const predSocket = usePredictionsSocket();

  const connect = useMutation({
    mutationFn: async () => (await http.post("/api/v1/mt5/connect")).data,
    onSuccess: () => {
      toast.success("MT5 connected");
      invalidateAll();
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });
  const disconnect = useMutation({
    mutationFn: async () => (await http.post("/api/v1/mt5/disconnect")).data,
    onSuccess: () => {
      toast.success("MT5 disconnected");
      invalidateAll();
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });
  const predict = useMutation({
    mutationFn: async () =>
      (await http.post("/api/v1/predictions/predict", { model_mode: "dual" })).data,
    onSuccess: () => toast.success("Prediction submitted"),
    onError: (e) => toast.error(getErrorMessage(e)),
  });
  const startAuto = useMutation({
    mutationFn: async () =>
      (await http.post("/api/v1/trading/auto/start", { interval: 1, model_mode: "dual" })).data,
    onSuccess: () => {
      toast.success("Auto trading started");
      invalidateAll();
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });
  const stopAuto = useMutation({
    mutationFn: async () => (await http.post("/api/v1/trading/auto/stop")).data,
    onSuccess: () => {
      toast.success("Auto trading stopped");
      invalidateAll();
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });

  return (
    <div className="mx-auto max-w-7xl">
      <div className="animate-in fade-in slide-in-from-bottom-6 duration-700">
        <PageHeader
          title="Dashboard"
          description="Operator overview · system health, runtime, quick actions"
        />
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          delay={100}
          icon={<Activity className="h-5 w-5" />}
          label="System Health"
          tone={health.isError ? "danger" : health.data?.status === "ok" ? "success" : "warning"}
          status={
            health.isLoading
              ? "…"
              : health.isError
                ? "ERROR"
                : (health.data?.status?.toUpperCase() ?? "—")
          }
          rows={
            health.data
              ? [
                ["Service", health.data.service],
                ["Env", health.data.environment],
                ["Version", health.data.version],
                ["Database", health.data.database],
              ]
              : [["Status", health.isError ? getErrorMessage(health.error) : "loading"]]
          }
        />
        <KpiCard
          delay={200}
          icon={<PlugZap className="h-5 w-5" />}
          label="MT5 Status"
          tone={mt5.isError ? "danger" : mt5.data?.connected ? "success" : "neutral"}
          status={mt5.isLoading ? "…" : mt5.data?.connected ? "CONNECTED" : "OFFLINE"}
          rows={
            mt5.data?.account_info
              ? Object.entries(mt5.data.account_info)
                .slice(0, 4)
                .map(([k, v]) => [k, String(v)])
              : [["Account", "—"]]
          }
        />
        <KpiCard
          delay={300}
          icon={<Gauge className="h-5 w-5" />}
          label="Auto Trading"
          tone={auto.isError ? "danger" : auto.data?.running ? "success" : "neutral"}
          status={auto.isLoading ? "…" : auto.data?.running ? "RUNNING" : "STOPPED"}
          rows={[
            ["Interval", auto.data?.interval ? `${auto.data.interval}s` : "—"],
            ["Mode", auto.data?.model_mode ?? "—"],
          ]}
        />
        <KpiCard
          delay={400}
          icon={<Cpu className="h-5 w-5" />}
          label="Models Loaded"
          tone={
            predSocket.status === "connected"
              ? "info"
              : predSocket.status === "reconnecting"
                ? "warning"
                : "neutral"
          }
          status={predSocket.status.toUpperCase()}
          rows={[
            ["H1", predSocket.last?.h1_model_loaded ? "loaded" : "—"],
            ["M5", predSocket.last?.m5_model_loaded ? "loaded" : "—"],
            ["Mode", predSocket.last?.current_mode ?? "—"],
          ]}
        />
      </div>

      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Trading Engine Control */}
        <div className="lg:col-span-2 animate-in fade-in slide-in-from-bottom-6 duration-700 delay-500 fill-mode-both">
          <PageSection title="Trading Engine Control" description="Manage the algorithmic trading runtime and model execution">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-2">
              <Button
                variant="default"
                onClick={() => startAuto.mutate()}
                disabled={startAuto.isPending}
                className="w-full h-12 transition-all active:scale-95 shadow-[0_0_15px_rgba(234,179,8,0.2)] hover:shadow-[0_0_25px_rgba(234,179,8,0.4)] text-base font-semibold"
              >
                <Play className="mr-2 h-5 w-5" /> {startAuto.isPending ? "Starting…" : "Start Auto"}
              </Button>
              <ConfirmAction
                variant="destructive"
                title="Stop Auto Trading?"
                description="The runtime will stop submitting trades immediately."
                confirmLabel="Stop"
                disabled={stopAuto.isPending}
                onConfirm={() => stopAuto.mutate()}
                className="w-full h-12 transition-all active:scale-95 shadow-[0_0_15px_rgba(239,68,68,0.2)] hover:shadow-[0_0_25px_rgba(239,68,68,0.4)] flex items-center justify-center gap-2 text-white text-base font-semibold"
              >
                <Power className="h-5 w-5" /> Stop Auto
              </ConfirmAction>
              <Button
                variant="secondary"
                onClick={() => predict.mutate()}
                disabled={predict.isPending}
                className="w-full h-12 transition-all active:scale-95 bg-white/5 hover:bg-white/10 text-white border border-white/10 text-base"
              >
                <Cpu className="mr-2 h-5 w-5" /> {predict.isPending ? "Predicting…" : "Run Predict"}
              </Button>
            </div>
          </PageSection>
        </div>

        {/* Platform Connections */}
        <div className="animate-in fade-in slide-in-from-bottom-6 duration-700 delay-600 fill-mode-both">
          <PageSection title="Platform Connections" description="Manage external platform bridges">
            <div className="flex flex-col gap-4 mt-2">
              <Button 
                onClick={() => connect.mutate()} 
                disabled={connect.isPending} 
                className="w-full transition-all active:scale-95 shadow-[0_0_15px_rgba(234,179,8,0.2)] hover:shadow-[0_0_25px_rgba(234,179,8,0.4)]"
              >
                <PlugZap className="mr-2 h-4 w-4" /> {connect.isPending ? "Connecting…" : "Connect MT5"}
              </Button>
              <ConfirmAction
                variant="outline"
                title="Disconnect MT5?"
                description="This will stop all MT5 communication until reconnected."
                confirmLabel="Disconnect"
                disabled={disconnect.isPending}
                onConfirm={() => disconnect.mutate()}
                className="w-full transition-all active:scale-95 border-white/10 hover:bg-white/5 bg-transparent text-white"
              >
                Disconnect MT5
              </ConfirmAction>
            </div>
          </PageSection>
        </div>
      </div>
    </div>
  );
}

import { cn } from "@/lib/utils";

function KpiCard({
  icon,
  label,
  status,
  tone,
  rows,
  delay = 0,
}: {
  icon: React.ReactNode;
  label: string;
  status: string;
  tone: "success" | "warning" | "danger" | "neutral" | "info";
  rows: [string, string | undefined][];
  delay?: number;
}) {
  return (
    <div 
      className="group relative flex flex-col h-full overflow-hidden rounded-xl border border-white/10 bg-slate-950/40 backdrop-blur-md p-5 transition-all duration-300 hover:bg-slate-900/60 hover:-translate-y-1 hover:border-yellow-500/30 hover:shadow-[0_8px_30px_-10px_rgba(234,179,8,0.15)] animate-in fade-in slide-in-from-bottom-6 duration-700 fill-mode-both"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-yellow-500/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

      {/* Top Row: Icon & Badge */}
      <div className="flex items-start justify-between mb-3">
        <div className="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-yellow-500/10 border border-yellow-500/20 text-yellow-500 transition-colors group-hover:text-yellow-400 group-hover:shadow-[0_0_15px_rgba(234,179,8,0.2)]">
          {icon}
        </div>
        <StatusBadge tone={tone}>{status}</StatusBadge>
      </div>

      {/* Label */}
      <div className="mb-5">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider">{label}</h3>
      </div>

      {/* Data Box */}
      <div className="mt-auto rounded-lg bg-black/40 p-3.5 border border-white/5">
        <dl className="space-y-2 font-mono text-xs">
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-center justify-between gap-3">
              <dt className="text-slate-400">{k}</dt>
              <dd className="font-semibold text-white truncate text-right">{v ?? "—"}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
