import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowLeft } from "lucide-react";
import { getModelArtifactPath } from "@/lib/backend-contract";
import { http, getErrorMessage } from "@/lib/http";
import { PageHeader, PageSection } from "@/components/operator/page-section";
import { StatusBadge } from "@/components/operator/status-badge";

type ModelDetail = {
  model_name: string;
  timeframe: string;
  version: string;
  is_active: boolean;
  created_at: string;
  metrics?: Record<string, unknown>;
  hyperparameters?: Record<string, unknown>;
  lookback?: number;
  epochs?: number;
  batch_size?: number;
  train_ratio?: number;
  scaler_path?: string | null;
  artifact_path?: string;
};

export const Route = createFileRoute("/models/$modelId")({
  head: () => ({ meta: [{ title: "Model Detail · Operator" }] }),
  component: ModelDetailPage,
});

function ModelDetailPage() {
  const { modelId } = Route.useParams();
  const q = useQuery<ModelDetail>({
    queryKey: ["model", modelId],
    queryFn: async () => (await http.get(`/api/v1/models/${modelId}`)).data,
    retry: 0,
  });
  const hyperparameters =
    q.data?.hyperparameters && Object.keys(q.data.hyperparameters).length > 0
      ? q.data.hyperparameters
      : {
          lookback: q.data?.lookback,
          epochs: q.data?.epochs,
          batch_size: q.data?.batch_size,
          train_ratio: q.data?.train_ratio,
        };

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <Link
        to="/models"
        className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-3 w-3" /> Back to models
      </Link>
      <PageHeader
        title={q.data?.model_name ?? "Model Detail"}
        description={`ID ${modelId}`}
        actions={
          q.data?.is_active ? (
            <StatusBadge tone="success">active</StatusBadge>
          ) : (
            <StatusBadge tone="neutral">idle</StatusBadge>
          )
        }
      />
      {q.isLoading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : q.isError ? (
        <p className="text-sm text-destructive">{getErrorMessage(q.error)}</p>
      ) : !q.data ? null : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <PageSection title="Basics">
            <DefList
              rows={[
                ["Name", q.data.model_name],
                ["Timeframe", q.data.timeframe],
                ["Version", q.data.version],
                ["Active", q.data.is_active ? "yes" : "no"],
                ["Created", q.data.created_at && new Date(q.data.created_at).toLocaleString()],
              ]}
            />
          </PageSection>
          <PageSection title="Metrics">
            <DefList rows={Object.entries(q.data.metrics ?? {}).map(([k, v]) => [k, fmt(v)])} />
          </PageSection>
          <PageSection title="Hyperparameters">
            <DefList
              rows={Object.entries(hyperparameters ?? {})
                .filter(([, v]) => v != null)
                .map(([k, v]) => [k, fmt(v)])}
            />
          </PageSection>
          <PageSection title="Artifacts">
            <DefList
              rows={[
                ["Model path", getModelArtifactPath(q.data)],
                ["Scaler path", q.data.scaler_path],
              ]}
            />
          </PageSection>
        </div>
      )}
    </div>
  );
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return String(v);
}

function DefList({ rows }: { rows: [string, React.ReactNode][] }) {
  if (!rows.length) return <p className="text-xs text-muted-foreground">No data</p>;
  return (
    <dl className="space-y-1.5 font-mono text-xs">
      {rows.map(([k, v]) => (
        <div
          key={k}
          className="flex items-start justify-between gap-3 border-b border-border/50 pb-1.5 last:border-0"
        >
          <dt className="text-muted-foreground">{k}</dt>
          <dd className="truncate text-right text-foreground">{v ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}
