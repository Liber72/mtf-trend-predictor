import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { toast } from "sonner";
import { LineChart, Play, TrendingDown, TrendingUp } from "lucide-react";
import {
  getCombinedPrediction,
  getPredictionProbability,
  getPredictionTimeframeDirection,
} from "@/lib/backend-contract";
import { http, getErrorMessage } from "@/lib/http";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { PageHeader, PageSection } from "@/components/operator/page-section";
import { StatusBadge, type StatusTone } from "@/components/operator/status-badge";
import { DataTable, EmptyState } from "./market-data";

type ModelMode = "dual" | "single_m5";

type PredictionResponse = {
  h1?: { direction: string; probability: number } | null;
  m5?: { direction: string; probability: number } | null;
  combined: { signal: string; confidence?: number | null; reason?: string | null };
  model_mode: ModelMode;
};

type PredictionHistoryItem = {
  predicted_at: string;
  model_mode: string;
  h1_direction?: string | null;
  h1_probability?: number | null;
  m5_direction?: string | null;
  m5_probability?: number | null;
  combined_signal?: string | null;
  combined_confidence?: number | null;
  trade_executed: boolean;
  reason?: string | null;
};

type PaginatedPredictions = {
  items: PredictionHistoryItem[];
  total: number;
  page: number;
  size: number;
  pages: number;
};

export const Route = createFileRoute("/predictions")({
  head: () => ({ meta: [{ title: "Predictions · Operator" }] }),
  component: PredictionsPage,
});

function PredictionsPage() {
  const qc = useQueryClient();
  const [mode, setMode] = useState<ModelMode>("dual");
  const [filterMode, setFilterMode] = useState<string>("all");
  const [page, setPage] = useState(1);
  const size = 10;
  const [last, setLast] = useState<PredictionResponse | null>(null);

  const history = useQuery<PaginatedPredictions>({
    queryKey: ["predictions", filterMode, page, size],
    queryFn: async () =>
      (
        await http.get("/api/v1/predictions", {
          params: {
            ...(filterMode !== "all" ? { model_mode: filterMode } : {}),
            page,
            size,
          },
        })
      ).data,
    retry: 0,
  });

  const predict = useMutation<PredictionResponse>({
    mutationFn: async () =>
      (await http.post("/api/v1/predictions/predict", { model_mode: mode })).data,
    onSuccess: (data) => {
      setLast(data);
      toast.success("Prediction received");
      qc.invalidateQueries({ queryKey: ["predictions"] });
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });

  const list = history.data?.items ?? [];
  const combined = getCombinedPrediction(last);

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader title="Predictions" description="Run inference, review history" />

      <PageSection
        title="Realtime Predict"
        actions={
          <div className="flex items-center gap-2">
            <Select value={mode} onValueChange={(value) => setMode(value as ModelMode)}>
              <SelectTrigger className="h-8 w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="dual">dual</SelectItem>
                <SelectItem value="single_m5">single_m5</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={() => predict.mutate()} disabled={predict.isPending}>
              <Play /> {predict.isPending ? "Predicting…" : "Run Predict"}
            </Button>
          </div>
        }
      >
        {!last ? (
          <EmptyState
            icon={<LineChart className="h-6 w-6" />}
            title="No prediction yet"
            hint="Click Run Predict to see model output."
          />
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <PredCard
              title="H1"
              direction={getPredictionTimeframeDirection(last, "h1") ?? undefined}
              prob={getPredictionProbability(last, "h1") ?? undefined}
            />
            <PredCard
              title="M5"
              direction={getPredictionTimeframeDirection(last, "m5") ?? undefined}
              prob={getPredictionProbability(last, "m5") ?? undefined}
            />
            <div className="rounded-lg border border-border bg-muted/40 p-4">
              <div className="text-xs uppercase tracking-widest text-muted-foreground">
                Combined
              </div>
              <div className="mt-2 flex items-center gap-2">
                <StatusBadge tone={signalTone(combined.signal ?? undefined)}>
                  {combined.signal ?? "—"}
                </StatusBadge>
                <span className="font-mono text-sm">
                  {combined.confidence != null ? (combined.confidence * 100).toFixed(1) + "%" : "—"}
                </span>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">{combined.reason ?? "—"}</p>
            </div>
          </div>
        )}
      </PageSection>

      <PageSection
        title="Prediction History"
        actions={
          <Select value={filterMode} onValueChange={setFilterMode}>
            <SelectTrigger className="h-8 w-36">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">all modes</SelectItem>
              <SelectItem value="dual">dual</SelectItem>
              <SelectItem value="single_m5">single_m5</SelectItem>
            </SelectContent>
          </Select>
        }
      >
        {history.isLoading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : history.isError ? (
          <p className="text-sm text-destructive">{getErrorMessage(history.error)}</p>
        ) : list.length === 0 ? (
          <EmptyState title="No predictions yet" hint="Run a prediction to populate history." />
        ) : (
          <>
            <DataTable
              head={[
                "When",
                "Mode",
                "H1",
                "H1 prob",
                "M5",
                "M5 prob",
                "Signal",
                "Conf",
                "Traded",
                "Reason",
              ]}
              rows={list.map((p) => [
                p.predicted_at ? new Date(p.predicted_at).toLocaleString() : "—",
                p.model_mode,
                p.h1_direction ?? "—",
                fmtPct(p.h1_probability),
                p.m5_direction ?? "—",
                fmtPct(p.m5_probability),
                <StatusBadge tone={signalTone(p.combined_signal)}>
                  {p.combined_signal ?? "—"}
                </StatusBadge>,
                fmtPct(p.combined_confidence),
                p.trade_executed ? (
                  <StatusBadge tone="success">yes</StatusBadge>
                ) : (
                  <StatusBadge tone="neutral">no</StatusBadge>
                ),
                <span className="text-muted-foreground">{p.reason ?? "—"}</span>,
              ])}
            />
            <div className="mt-3 flex items-center justify-end gap-2 text-xs">
              <Button
                variant="outline"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                Prev
              </Button>
              <span className="text-muted-foreground">Page {page}</span>
              <Button
                variant="outline"
                size="sm"
                disabled={list.length < size}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </>
        )}
      </PageSection>
    </div>
  );
}

function PredCard({
  title,
  direction,
  prob,
}: {
  title: string;
  direction?: string;
  prob?: number;
}) {
  const upper = direction?.toUpperCase();
  const isUp = upper === "UP" || upper === "BUY";
  const isDown = upper === "DOWN" || upper === "SELL";
  const tone = isUp ? "success" : isDown ? "danger" : "neutral";
  const Icon = isUp ? TrendingUp : isDown ? TrendingDown : LineChart;
  return (
    <div className="rounded-lg border border-border bg-muted/40 p-4">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-widest text-muted-foreground">{title}</div>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <div className="mt-2 flex items-center gap-2">
        <StatusBadge tone={tone}>{direction ?? "—"}</StatusBadge>
        <span className="font-mono text-sm">{fmtPct(prob)}</span>
      </div>
    </div>
  );
}

function fmtPct(n?: number | null) {
  return typeof n === "number" ? (n * 100).toFixed(1) + "%" : "—";
}
function signalTone(s?: string | null): StatusTone {
  if (!s) return "neutral";
  const u = s.toUpperCase();
  if (u === "BUY") return "success";
  if (u === "SELL") return "danger";
  if (u === "HOLD" || u === "WAIT") return "warning";
  return "info";
}
