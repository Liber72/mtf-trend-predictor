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

export const Route = createFileRoute("/")({
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
      <PageHeader
        title="Dashboard"
        description="Operator overview · system health, runtime, quick actions"
      />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          icon={<Activity className="h-4 w-4" />}
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
          icon={<PlugZap className="h-4 w-4" />}
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
          icon={<Gauge className="h-4 w-4" />}
          label="Auto Trading"
          tone={auto.isError ? "danger" : auto.data?.running ? "success" : "neutral"}
          status={auto.isLoading ? "…" : auto.data?.running ? "RUNNING" : "STOPPED"}
          rows={[
            ["Interval", auto.data?.interval ? `${auto.data.interval}s` : "—"],
            ["Mode", auto.data?.model_mode ?? "—"],
          ]}
        />
        <KpiCard
          icon={<Cpu className="h-4 w-4" />}
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

      <div className="mt-6">
        <PageSection title="Quick Actions" description="One-click control plane for the runtime">
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => connect.mutate()} disabled={connect.isPending}>
              <PlugZap /> {connect.isPending ? "Connecting…" : "Connect MT5"}
            </Button>
            <ConfirmAction
              variant="outline"
              title="Disconnect MT5?"
              description="This will stop all MT5 communication until reconnected."
              confirmLabel="Disconnect"
              disabled={disconnect.isPending}
              onConfirm={() => disconnect.mutate()}
            >
              Disconnect MT5
            </ConfirmAction>
            <Button
              variant="secondary"
              onClick={() => predict.mutate()}
              disabled={predict.isPending}
            >
              <Play /> {predict.isPending ? "Predicting…" : "Run Predict"}
            </Button>
            <Button
              variant="default"
              onClick={() => startAuto.mutate()}
              disabled={startAuto.isPending}
            >
              <Play /> {startAuto.isPending ? "Starting…" : "Start Auto Trading"}
            </Button>
            <ConfirmAction
              variant="destructive"
              title="Stop Auto Trading?"
              description="The runtime will stop submitting trades immediately."
              confirmLabel="Stop"
              disabled={stopAuto.isPending}
              onConfirm={() => stopAuto.mutate()}
            >
              <Power /> Stop Auto Trading
            </ConfirmAction>
          </div>
        </PageSection>
      </div>
    </div>
  );
}

function KpiCard({
  icon,
  label,
  status,
  tone,
  rows,
}: {
  icon: React.ReactNode;
  label: string;
  status: string;
  tone: "success" | "warning" | "danger" | "neutral" | "info";
  rows: [string, string | undefined][];
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-muted-foreground">
          <span className="text-primary">{icon}</span>
          {label}
        </div>
        <StatusBadge tone={tone}>{status}</StatusBadge>
      </div>
      <dl className="mt-3 space-y-1.5 font-mono text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between gap-2">
            <dt className="truncate text-muted-foreground">{k}</dt>
            <dd className="truncate text-foreground">{v ?? "—"}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
