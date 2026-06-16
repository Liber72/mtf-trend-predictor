import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient, keepPreviousData } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { toast } from "sonner";
import { PlugZap, Play, Power, Trash2, ChevronLeft, ChevronRight } from "lucide-react";
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
  volume?: number | null;
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
  total: number;
  page: number;
  size: number;
  pages: number;
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
  const [tradePage, setTradePage] = useState(1);
  const trades = useQuery<PaginatedTrades>({
    queryKey: ["trades", tradePage],
    queryFn: async () => (await http.get(`/api/v1/trades?page=${tradePage}&size=10`)).data,
    placeholderData: keepPreviousData,
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
  const [volume, setVolume] = useState("0.1");
  const [autoMode, setAutoMode] = useState<ModelMode>("dual");
  const start = useMutation({
    mutationFn: async () =>
      (
        await http.post("/api/v1/trading/auto/start", {
          interval: Number(interval),
          model_mode: autoMode,
          volume: Number(volume),
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

  useEffect(() => {
    if (autoStatus.data?.running) {
      if (autoStatus.data.interval) {
        setInterval(autoStatus.data.interval.toString());
      }
      if (autoStatus.data.volume) {
        setVolume(autoStatus.data.volume.toString());
      }
      if (autoStatus.data.model_mode) {
        setAutoMode(autoStatus.data.model_mode as ModelMode);
      }
    }
  }, [autoStatus.data]);

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
          <Field label="Volume (Lot)">
            <Input type="number" step="0.01" min="0.01" value={volume} onChange={(e) => setVolume(e.target.value)} />
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
          <Button onClick={() => start.mutate()} disabled={start.isPending || autoStatus.data?.running}>
            <Play /> {start.isPending ? "Starting…" : "Start"}
          </Button>
          <ConfirmAction
            title="Stop Auto Trading?"
            description="The runtime will stop submitting trades immediately."
            confirmLabel="Stop"
            variant="destructive"
            onConfirm={() => stop.mutate()}
            disabled={stop.isPending || !autoStatus.data?.running}
          >
            <Power /> Stop
          </ConfirmAction>
        </div>
        
        <AutoTradingTerminal isRunning={autoStatus.data?.running ?? false} />
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
          <div className="space-y-4">
            <DataTable
              head={["ID", "Symbol", "Direction", "Volume", "Status", "Opened (Server)", "Profit"]}
              rows={tradeList.map((t) => [
                <span className="text-muted-foreground font-mono">{t.id}</span>,
                t.symbol,
                t.direction,
                t.volume,
                <StatusBadge tone={statusTone(t.status)}>{t.status ?? "—"}</StatusBadge>,
                getTradeOpenedAt(t) ? new Date(getTradeOpenedAt(t) as string).toLocaleString('en-GB', { timeZone: 'UTC' }) : "—",
                <span className={(getTradePnl(t) ?? 0) >= 0 ? "text-success" : "text-destructive"}>
                  {typeof getTradePnl(t) === "number" ? (getTradePnl(t) as number).toFixed(2) : "—"}
                </span>,
              ])}
            />
            {trades.data && trades.data.pages > 1 && (
              <div className="flex items-center justify-end gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={tradePage === 1}
                  onClick={() => setTradePage((p) => p - 1)}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="text-sm text-muted-foreground">
                  Page {trades.data.page} of {trades.data.pages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={tradePage === trades.data.pages}
                  onClick={() => setTradePage((p) => p + 1)}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
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

type LogEntry = {
  time: string;
  type: string;
  message: any;
};

function AutoTradingTerminal({ isRunning }: { isRunning: boolean }) {
  const { data } = useQuery<LogEntry[]>({
    queryKey: ["auto-trade-logs"],
    queryFn: async () => (await http.get("/api/v1/trading/auto/logs")).data,
    refetchInterval: isRunning ? 1000 : 5000,
  });

  const qc = useQueryClient();
  const clearLogs = useMutation({
    mutationFn: async () => (await http.delete("/api/v1/trading/auto/logs")).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["auto-trade-logs"] });
    },
  });

  if (!data || data.length === 0) return null;

  return (
    <div className="mt-4 flex h-[28rem] flex-col rounded-xl border border-border bg-muted/20 p-4 font-sans text-sm shadow-inner">
      <div className="mb-4 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            {isRunning && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>}
            <span className={`relative inline-flex h-2 w-2 rounded-full ${isRunning ? 'bg-emerald-500' : 'bg-slate-500'}`}></span>
          </span>
          <span className="font-semibold uppercase tracking-wider text-foreground">
            Live Auto Trading Feed
          </span>
        </div>
        <Button variant="ghost" size="icon" onClick={() => clearLogs.mutate()} disabled={clearLogs.isPending} title="Clear Logs">
          <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
        </Button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto pr-2">
        {data.slice().reverse().map((log, i) => {
          if (log.type === "prediction") {
            const p = log.message;
            return (
              <div key={i} className="flex flex-col gap-2 rounded-lg border border-border bg-background p-3 shadow-sm">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">{log.time}</span>
                  <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium uppercase text-muted-foreground">
                    {p.model_mode}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-xs md:grid-cols-3">
                  {p.model_mode === 'dual' && (
                    <div className="flex flex-col rounded bg-muted/50 p-2">
                      <span className="text-muted-foreground">H1</span>
                      <div className="flex items-center gap-1.5">
                        <span className={`font-semibold ${p.h1_dir === 'UP' || p.h1_dir === 'BUY' ? 'text-emerald-500' : p.h1_dir === 'DOWN' || p.h1_dir === 'SELL' ? 'text-red-500' : ''}`}>{p.h1_dir}</span>
                        <span className="text-muted-foreground">{(p.h1_prob * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  )}
                  <div className="flex flex-col rounded bg-muted/50 p-2">
                    <span className="text-muted-foreground">M5</span>
                    <div className="flex items-center gap-1.5">
                      <span className={`font-semibold ${p.m5_dir === 'UP' || p.m5_dir === 'BUY' ? 'text-emerald-500' : p.m5_dir === 'DOWN' || p.m5_dir === 'SELL' ? 'text-red-500' : ''}`}>{p.m5_dir}</span>
                      <span className="text-muted-foreground">{(p.m5_prob * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  <div className="flex flex-col rounded bg-muted/50 p-2">
                    <span className="text-muted-foreground">Combined</span>
                    <div className="flex items-center gap-1.5">
                      <span className={`font-semibold ${p.signal === 'BUY' ? 'text-emerald-500' : p.signal === 'SELL' ? 'text-red-500' : 'text-yellow-500'}`}>{p.signal}</span>
                      {p.confidence && <span className="text-muted-foreground">{(p.confidence * 100).toFixed(1)}%</span>}
                    </div>
                  </div>
                </div>
                
                <div className={`mt-2 rounded-md px-3 py-2 text-xs font-medium ${
                  p.action_type === 'success' ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400' : 
                  p.action_type === 'warning' ? 'bg-yellow-500/15 text-yellow-600 dark:text-yellow-400' : 
                  'bg-blue-500/15 text-blue-600 dark:text-blue-400'
                }`}>
                  {p.action_type === 'success' ? 'VÀO LỆNH: ' : p.action_type === 'warning' ? 'WAIT: ' : 'BỎ QUA: '}{p.action}
                </div>
              </div>
            );
          }
          
          return (
            <div
              key={i}
              className={`flex items-start gap-3 rounded-lg border border-border bg-background p-3 shadow-sm ${
                log.type === "error"
                  ? "text-red-500"
                  : log.type === "success"
                    ? "text-emerald-500"
                    : "text-foreground"
              }`}
            >
              <span className="shrink-0 text-xs text-muted-foreground pt-0.5">[{log.time}]</span>
              <span className="whitespace-pre-wrap text-sm">{typeof log.message === 'string' ? log.message : JSON.stringify(log.message)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
