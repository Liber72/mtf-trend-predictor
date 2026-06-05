import { createFileRoute } from "@tanstack/react-router";
import { Radio } from "lucide-react";
import { PageHeader, PageSection } from "@/components/operator/page-section";
import { StatusBadge, type StatusTone } from "@/components/operator/status-badge";
import { useTradesSocket } from "@/hooks/use-trades-socket";
import { usePredictionsSocket } from "@/hooks/use-predictions-socket";
import type { SocketStatus } from "@/hooks/use-websocket";

function getEventType(event: unknown): string | null {
  if (!event || typeof event !== "object") return null;
  const value = (event as Record<string, unknown>).type;
  return typeof value === "string" ? value : null;
}

export const Route = createFileRoute("/monitor")({
  head: () => ({ meta: [{ title: "Monitor · Operator" }] }),
  component: MonitorPage,
});

function MonitorPage() {
  const trades = useTradesSocket();
  const preds = usePredictionsSocket();

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader title="Monitor" description="Live runtime streams from WebSocket" />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <StreamPanel
          title="Trade Stream"
          path="/api/v1/ws/trades"
          status={trades.status}
          events={trades.events}
        />

        <PageSection
          title="Prediction Status Stream"
          description="WS /api/v1/ws/predictions"
          actions={<StatusBadge tone={statusTone(preds.status)}>{preds.status}</StatusBadge>}
        >
          <div className="grid grid-cols-3 gap-3 font-mono text-xs">
            <Tile
              label="H1 model"
              value={preds.last?.h1_model_loaded ? "loaded" : "—"}
              ok={!!preds.last?.h1_model_loaded}
            />
            <Tile
              label="M5 model"
              value={preds.last?.m5_model_loaded ? "loaded" : "—"}
              ok={!!preds.last?.m5_model_loaded}
            />
            <Tile
              label="Mode"
              value={preds.last?.current_mode ?? "—"}
              ok={!!preds.last?.current_mode}
            />
          </div>
          <EventLog events={preds.events} />
        </PageSection>
      </div>
    </div>
  );
}

function StreamPanel({
  title,
  path,
  status,
  events,
}: {
  title: string;
  path: string;
  status: SocketStatus;
  events: unknown[];
}) {
  return (
    <PageSection
      title={title}
      description={`WS ${path}`}
      actions={<StatusBadge tone={statusTone(status)}>{status}</StatusBadge>}
    >
      {events.length === 0 ? (
        <div className="flex flex-col items-center gap-2 py-8 text-center">
          <Radio className="h-6 w-6 text-muted-foreground" />
          <p className="text-sm font-medium">Waiting for events…</p>
          <p className="text-xs text-muted-foreground">Buffer keeps the latest 200 frames.</p>
        </div>
      ) : (
        <EventLog events={events} />
      )}
    </PageSection>
  );
}

function EventLog({ events }: { events: unknown[] }) {
  if (events.length === 0) return null;
  const recent = [...events].slice(-100).reverse();
  return (
    <div className="mt-4 max-h-96 overflow-auto rounded-md border border-border bg-background/40 font-mono text-[11px]">
      {recent.map((e, i) => (
        <div key={i} className="flex gap-3 border-b border-border/60 px-3 py-1.5 last:border-0">
          <span className="shrink-0 text-muted-foreground">{new Date().toLocaleTimeString()}</span>
          {getEventType(e) && (
            <StatusBadge tone="info" dot={false} className="!py-0">
              {getEventType(e)}
            </StatusBadge>
          )}
          <pre className="min-w-0 flex-1 truncate text-foreground">{JSON.stringify(e)}</pre>
        </div>
      ))}
    </div>
  );
}

function Tile({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="rounded-md border border-border bg-muted/40 p-3">
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className={`mt-1 ${ok ? "text-success" : "text-muted-foreground"}`}>{value}</div>
    </div>
  );
}

function statusTone(s: SocketStatus): StatusTone {
  if (s === "connected") return "success";
  if (s === "reconnecting" || s === "connecting") return "warning";
  return "danger";
}
