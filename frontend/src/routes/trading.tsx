import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { toast } from "sonner";
import { PlugZap, Play, Power } from "lucide-react";
import { getTradeOpenedAt, getTradePnl } from "@/lib/backend-contract";
import { http, getErrorMessage } from "@/lib/http";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { PageHeader, PageSection } from "@/components/operator/page-section";
import { StatusBadge, type StatusTone } from "@/components/operator/status-badge";
import { ConfirmAction } from "@/components/operator/confirm-action";
import { DataTable, EmptyState, Field } from "./market-data";

type ModelMode = "dual" | "single_m5";

type Mt5Status = {
  connected: boolean;
  account_info?: Record<string, string | number | boolean | null> | null;
};

type Position = {
  ticket: number;
  type: string;
  symbol: string;
  volume: number;
  price_open: number;
  price_current: number;
  sl: number;
  tp: number;
  profit: number;
};

type AutoTradeStatus = {
  running: boolean;
  interval?: number | null;
  model_mode?: string | null;
};

type TradeItem = {
  id: number;
  symbol: string;
  direction: string;
  volume: number;
  status: string;
  entry_time?: string | null;
  pnl?: number | null;
};

type PaginatedTrades = {
  items: TradeItem[];
};

export const Route = createFileRoute("/trading")({
  head: () => ({ meta: [{ title: "Trading · Operator" }] }),
  component: TradingPage,
});

function TradingPage() {
  const qc = useQueryClient();

  const mt5 = useQuery<Mt5Status>({
    queryKey: ["mt5-status"],
    queryFn: async () => (await http.get("/api/v1/mt5/status")).data,
    retry: 0,
  });
  const positions = useQuery<Position[]>({
    queryKey: ["mt5-positions"],
    queryFn: async () => (await http.get("/api/v1/mt5/positions")).data,
    retry: 0,
  });
  const autoStatus = useQuery<AutoTradeStatus>({
    queryKey: ["auto-trade-status"],
    queryFn: async () => (await http.get("/api/v1/trading/auto/status")).data,
    retry: 0,
  });
  const trades = useQuery<PaginatedTrades>({
    queryKey: ["trades"],
    queryFn: async () => (await http.get("/api/v1/trades")).data,
    retry: 0,
  });

  const invMt5 = () => {
    qc.invalidateQueries({ queryKey: ["mt5-status"] });
    qc.invalidateQueries({ queryKey: ["mt5-positions"] });
  };

  const connect = useMutation({
    mutationFn: async () => (await http.post("/api/v1/mt5/connect")).data,
    onSuccess: () => {
      toast.success("Connected");
      invMt5();
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });
  const disconnect = useMutation({
    mutationFn: async () => (await http.post("/api/v1/mt5/disconnect")).data,
    onSuccess: () => {
      toast.success("Disconnected");
      invMt5();
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });

  // Manual trade
  const [signal, setSignal] = useState<"BUY" | "SELL">("BUY");
  const [confidence, setConfidence] = useState("0.75");
  const execute = useMutation({
    mutationFn: async () =>
      (await http.post("/api/v1/trading/execute", { signal, confidence: Number(confidence) })).data,
    onSuccess: (d) => {
      toast.success(d?.message ?? "Trade submitted");
      qc.invalidateQueries({ queryKey: ["trades"] });
      qc.invalidateQueries({ queryKey: ["mt5-positions"] });
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });

  // Auto trade
  const [interval, setInterval] = useState("1");
  const [autoMode, setAutoMode] = useState<ModelMode>("dual");
  const start = useMutation({
    mutationFn: async () =>
      (
        await http.post("/api/v1/trading/auto/start", {
          interval: Number(interval),
          model_mode: autoMode,
        })
      ).data,
    onSuccess: () => {
      toast.success("Auto started");
      qc.invalidateQueries({ queryKey: ["auto-trade-status"] });
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });
  const stop = useMutation({
    mutationFn: async () => (await http.post("/api/v1/trading/auto/stop")).data,
    onSuccess: () => {
      toast.success("Auto stopped");
      qc.invalidateQueries({ queryKey: ["auto-trade-status"] });
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });

  const positionList = positions.data ?? [];
  const tradeList = trades.data?.items ?? [];

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader title="Trading" description="MT5 connection, positions, manual + auto trading" />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <PageSection
          title="MT5 Connection"
          actions={
            <StatusBadge tone={mt5.data?.connected ? "success" : "neutral"}>
              {mt5.data?.connected ? "connected" : "offline"}
            </StatusBadge>
          }
        >
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => connect.mutate()} disabled={connect.isPending}>
              <PlugZap /> {connect.isPending ? "Connecting…" : "Connect"}
            </Button>
            <ConfirmAction
              title="Disconnect MT5?"
              description="The runtime will lose access to MT5 until reconnected."
              confirmLabel="Disconnect"
              variant="outline"
              onConfirm={() => disconnect.mutate()}
              disabled={disconnect.isPending}
            >
              Disconnect
            </ConfirmAction>
          </div>
          {mt5.data?.account_info && (
            <div className="mt-4 grid grid-cols-2 gap-2 rounded-md border border-border bg-muted/40 p-3 font-mono text-xs">
              {Object.entries(mt5.data.account_info)
                .slice(0, 8)
                .map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2">
                    <span className="text-muted-foreground">{k}</span>
                    <span>{String(v)}</span>
                  </div>
                ))}
            </div>
          )}
        </PageSection>

        <PageSection title="Manual Trade" description="Submit a discretionary order">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Signal">
              <Select value={signal} onValueChange={(value) => setSignal(value as "BUY" | "SELL")}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="BUY">BUY</SelectItem>
                  <SelectItem value="SELL">SELL</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            <Field label="Confidence (0-1)">
              <Input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={confidence}
                onChange={(e) => setConfidence(e.target.value)}
              />
            </Field>
          </div>
          <ConfirmAction
            className="mt-4"
            title={`Execute ${signal}?`}
            description={`Submit a ${signal} order with confidence ${confidence}.`}
            confirmLabel="Execute"
            variant="default"
            onConfirm={() => execute.mutate()}
            disabled={execute.isPending}
          >
            <Play /> {execute.isPending ? "Executing…" : "Execute Trade"}
          </ConfirmAction>
        </PageSection>
      </div>

      <PageSection
        title="Auto Trading"
        actions={
          <StatusBadge tone={autoStatus.data?.running ? "success" : "neutral"}>
            {autoStatus.data?.running
              ? `running · ${autoStatus.data.interval}s · ${autoStatus.data.model_mode}`
              : "stopped"}
          </StatusBadge>
        }
      >
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Field label="Interval (s)">
            <Input type="number" value={interval} onChange={(e) => setInterval(e.target.value)} />
          </Field>
          <Field label="Model mode">
            <Select value={autoMode} onValueChange={(value) => setAutoMode(value as ModelMode)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="dual">dual</SelectItem>
                <SelectItem value="single_m5">single_m5</SelectItem>
              </SelectContent>
            </Select>
          </Field>
        </div>
        <div className="mt-4 flex gap-2">
          <Button onClick={() => start.mutate()} disabled={start.isPending}>
            <Play /> {start.isPending ? "Starting…" : "Start"}
          </Button>
          <ConfirmAction
            title="Stop Auto Trading?"
            description="The runtime will stop submitting trades immediately."
            confirmLabel="Stop"
            variant="destructive"
            onConfirm={() => stop.mutate()}
            disabled={stop.isPending}
          >
            <Power /> Stop
          </ConfirmAction>
        </div>
      </PageSection>

      <PageSection
        title="Open Positions"
        actions={
          <Button variant="outline" size="sm" onClick={() => positions.refetch()}>
            Refresh
          </Button>
        }
      >
        {positions.isLoading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : positions.isError ? (
          <p className="text-sm text-destructive">{getErrorMessage(positions.error)}</p>
        ) : positionList.length === 0 ? (
          <EmptyState
            title="No open positions"
            hint="Positions will appear here once trades are open."
          />
        ) : (
          <DataTable
            head={["Ticket", "Type", "Symbol", "Vol", "Open", "Current", "SL", "TP", "P/L"]}
            rows={positionList.map((p) => [
              p.ticket,
              p.type,
              p.symbol,
              p.volume,
              p.price_open,
              p.price_current,
              p.sl,
              p.tp,
              <span className={p.profit >= 0 ? "text-success" : "text-destructive"}>
                {p.profit}
              </span>,
            ])}
          />
        )}
      </PageSection>

      <PageSection title="Trade History">
        {trades.isLoading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : trades.isError ? (
          <p className="text-sm text-destructive">{getErrorMessage(trades.error)}</p>
        ) : tradeList.length === 0 ? (
          <EmptyState title="No trades yet" hint="Executed trades will appear here." />
        ) : (
          <DataTable
            head={["ID", "Symbol", "Direction", "Volume", "Status", "Opened", "P/L"]}
            rows={tradeList.map((t) => [
              <span className="text-muted-foreground">{String(t.id).slice(0, 8)}</span>,
              t.symbol,
              t.direction,
              t.volume,
              <StatusBadge tone={statusTone(t.status)}>{t.status ?? "—"}</StatusBadge>,
              getTradeOpenedAt(t) ? new Date(getTradeOpenedAt(t) as string).toLocaleString() : "—",
              <span className={(getTradePnl(t) ?? 0) >= 0 ? "text-success" : "text-destructive"}>
                {getTradePnl(t) ?? "—"}
              </span>,
            ])}
          />
        )}
      </PageSection>
    </div>
  );
}

function statusTone(s?: string): StatusTone {
  if (!s) return "neutral";
  const u = s.toLowerCase();
  if (u.includes("open")) return "info";
  if (u.includes("close") || u.includes("done")) return "success";
  if (u.includes("fail") || u.includes("error") || u.includes("reject")) return "danger";
  return "neutral";
}
