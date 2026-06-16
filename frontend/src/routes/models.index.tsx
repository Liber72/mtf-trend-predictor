import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { toast } from "sonner";
import { Boxes, Play } from "lucide-react";
import { http, getErrorMessage } from "@/lib/http";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PageHeader, PageSection } from "@/components/operator/page-section";
import { StatusBadge } from "@/components/operator/status-badge";
import { DataTable, EmptyState, Field } from "./market-data";

type ModelMetrics = Partial<{
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
}>;

type ModelVersion = {
  id: number;
  model_name: string;
  timeframe: string;
  version: string;
  metrics: ModelMetrics;
  is_active: boolean;
  created_at: string;
};

type PaginatedModels = {
  items: ModelVersion[];
  total: number;
  page: number;
  size: number;
  pages: number;
};

type TrainPayload = {
  timeframe: string;
  lookback: number;
  epochs: number;
  batch_size: number;
  train_ratio: number;
};

type TrainResponse = {
  timeframe: string;
  model_path: string;
  metrics?: ModelMetrics;
  message: string;
};

export const Route = createFileRoute("/models/")({
  head: () => ({ meta: [{ title: "Models · Operator" }] }),
  component: ModelsPage,
});

function ModelsPage() {
  const qc = useQueryClient();
  const [filterTf, setFilterTf] = useState("");
  const [page, setPage] = useState(1);
  const size = 20;

  const models = useQuery<PaginatedModels>({
    queryKey: ["models", { tf: filterTf, page, size }],
    queryFn: async () =>
      (
        await http.get("/api/v1/models", {
          params: { timeframe: filterTf || undefined, page, size },
        })
      ).data,
    retry: 0,
  });

  const train = useMutation<TrainResponse, Error, TrainPayload>({
    mutationFn: async (payload) => (await http.post("/api/v1/models/train", payload, { timeout: 3600000 })).data,
    onSuccess: (data) => {
      toast.success(
        `Trained ${data?.timeframe ?? "model"} successfully`,
      );
      qc.invalidateQueries({ queryKey: ["models"] });
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });

  const activate = useMutation({
    mutationFn: async (id: string) => (await http.patch(`/api/v1/models/${id}/activate`)).data,
    onSuccess: () => {
      toast.success("Model activated");
      qc.invalidateQueries({ queryKey: ["models"] });
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });

  const list = models.data?.items ?? [];

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader title="Models" description="Train, version, and activate prediction models" />
      <TrainForm pending={train.isPending} onSubmit={(p) => train.mutate(p)} result={train.data} />

      <PageSection
        title="Model Versions"
        description="All trained model versions"
        actions={
          <div className="flex items-center gap-2">
            <Input
              value={filterTf}
              onChange={(e) => {
                setFilterTf(e.target.value);
                setPage(1);
              }}
              placeholder="Filter timeframe"
              className="h-8 w-36"
            />
            <Button variant="outline" size="sm" onClick={() => models.refetch()}>
              Refresh
            </Button>
          </div>
        }
      >
        {models.isLoading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : models.isError ? (
          <p className="text-sm text-destructive">{getErrorMessage(models.error)}</p>
        ) : list.length === 0 ? (
          <EmptyState
            icon={<Boxes className="h-6 w-6" />}
            title="No models yet"
            hint="Train your first model with the form above."
          />
        ) : (
          <>
            <DataTable
              head={["ID", "Name", "TF", "Version", "Status", "Acc", "F1", "Created", "Actions"]}
              rows={list.map((m) => [
                <span className="text-muted-foreground">{String(m.id).slice(0, 8)}</span>,
                m.model_name,
                m.timeframe,
                m.version,
                m.is_active ? (
                  <StatusBadge tone="success">active</StatusBadge>
                ) : (
                  <StatusBadge tone="neutral">idle</StatusBadge>
                ),
                m.metrics?.accuracy?.toFixed?.(3) ?? "—",
                m.metrics?.f1_score?.toFixed?.(3) ?? "—",
                m.created_at ? new Date(m.created_at).toLocaleString() : "—",
                <div className="flex gap-2">
                  <Link
                    to="/models/$modelId"
                    params={{ modelId: String(m.id) }}
                    className="text-primary hover:underline"
                  >
                    View
                  </Link>
                  {!m.is_active && (
                    <button
                      onClick={() => activate.mutate(String(m.id))}
                      className="text-success hover:underline disabled:opacity-50"
                      disabled={activate.isPending}
                    >
                      Activate
                    </button>
                  )}
                </div>,
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

function TrainForm({
  pending,
  onSubmit,
  result,
}: {
  pending: boolean;
  onSubmit: (payload: TrainPayload) => void;
  result?: TrainResponse;
}) {
  const [timeframe, setTimeframe] = useState("H1");
  const [lookback, setLookback] = useState("60");
  const [epochs, setEpochs] = useState("20");
  const [batchSize, setBatchSize] = useState("64");
  const [trainRatio, setTrainRatio] = useState("0.8");

  return (
    <PageSection title="Train Model" description="Kick off a new training run">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Field label="Timeframe">
          <Input value={timeframe} onChange={(e) => setTimeframe(e.target.value)} />
        </Field>
        <Field label="Lookback">
          <Input type="number" value={lookback} onChange={(e) => setLookback(e.target.value)} />
        </Field>
        <Field label="Epochs">
          <Input type="number" value={epochs} onChange={(e) => setEpochs(e.target.value)} />
        </Field>
        <Field label="Batch size">
          <Input type="number" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
        </Field>
        <Field label="Train ratio">
          <Input
            type="number"
            step="0.05"
            value={trainRatio}
            onChange={(e) => setTrainRatio(e.target.value)}
          />
        </Field>
      </div>
      <Button
        className="mt-4"
        disabled={pending}
        onClick={() =>
          onSubmit({
            timeframe,
            lookback: Number(lookback),
            epochs: Number(epochs),
            batch_size: Number(batchSize),
            train_ratio: Number(trainRatio),
          })
        }
      >
        <Play /> {pending ? "Waiting for model..." : "Train Model"}
      </Button>
      {result && (
        <div className="mt-4 grid grid-cols-2 gap-3 rounded-md border border-border bg-white/[0.02] p-3 font-mono text-xs md:grid-cols-4 shadow-sm backdrop-blur-xl">
          <Stat label="accuracy" value={result.metrics?.accuracy} />
          <Stat label="precision" value={result.metrics?.precision} />
          <Stat label="recall" value={result.metrics?.recall} />
          <Stat label="f1" value={result.metrics?.f1_score} />
          <div className="col-span-full text-muted-foreground">{result.model_path}</div>
        </div>
      )}
      
      {/* Hiển thị Terminal độc lập với frontend state (dựa vào backend state) */}
      <div className="flex flex-col gap-4">
        <LiveTrainingTerminal timeframe="H1" />
        <LiveTrainingTerminal timeframe="M5" />
      </div>
    </PageSection>
  );
}

function LiveTrainingTerminal({ timeframe }: { timeframe: string }) {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ["train-status", timeframe],
    queryFn: async () => (await http.get(`/api/v1/models/train-status?timeframe=${timeframe}`)).data,
    refetchInterval: 1000,
  });

  const cancel = useMutation({
    mutationFn: async () => (await http.post(`/api/v1/models/cancel-train?timeframe=${timeframe}`)).data,
    onSuccess: () => {
      toast.success(`Successfully cancelled training for ${timeframe}`);
      qc.invalidateQueries({ queryKey: ["train-status"] });
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });

  if (!data || (data.status !== "training" && data.status !== "processing")) return null;

  return (
    <div className="mt-4 rounded-xl border border-white/10 bg-black/40 p-4 font-mono text-xs shadow-inner backdrop-blur-md relative">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-yellow-500"></span>
          </span>
          <span className="text-yellow-500 font-semibold uppercase tracking-wider">
            [{timeframe}] {data.status === "training" ? "Training in progress..." : "Processing..."}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-slate-400">
            Epoch {data.epoch} / {data.total_epochs}
          </div>
          {data.status === "training" && (
            <Button
              variant="destructive"
              size="sm"
              className="h-6 text-[10px] px-2 py-0 uppercase tracking-widest font-bold"
              disabled={cancel.isPending}
              onClick={() => cancel.mutate()}
            >
              Cancel
            </Button>
          )}
        </div>
      </div>

      <div className="space-y-1 mt-3">
        {data.logs?.length > 0 ? (
          data.logs.map((log: any, i: number) => (
            <div key={i} className="flex flex-col md:flex-row md:items-center justify-between text-slate-300 border-b border-white/5 pb-1">
              <span className="text-slate-500">Epoch {log.epoch}</span>
              <div className="flex gap-4">
                <span>loss: <span className="text-rose-400">{log.loss.toFixed(4)}</span></span>
                <span>acc: <span className="text-emerald-400">{log.accuracy.toFixed(4)}</span></span>
                <span>val_loss: <span className="text-rose-400">{log.val_loss.toFixed(4)}</span></span>
                <span>val_acc: <span className="text-emerald-400">{log.val_accuracy.toFixed(4)}</span></span>
              </div>
            </div>
          ))
        ) : (
          <div className="text-slate-500 animate-pulse">Initializing model and dataset...</div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value?: number }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className="text-success">{typeof value === "number" ? value.toFixed(4) : "—"}</div>
    </div>
  );
}
