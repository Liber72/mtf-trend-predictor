import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { toast } from "sonner";
import { Download, Upload, FileSpreadsheet } from "lucide-react";
import { http, getErrorMessage } from "@/lib/http";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PageHeader, PageSection } from "@/components/operator/page-section";
import { StatusBadge } from "@/components/operator/status-badge";

type MarketDataFile = {
  filename: string;
  size_mb: number;
  path: string;
};

type CrawlTimeframeResult = {
  timeframe: string;
  rows: number;
  status: string;
};

type CrawlResponse = {
  symbol: string;
  results: CrawlTimeframeResult[];
};

type ImportCsvResult = {
  rows_imported: number;
  symbol: string;
  timeframe: string;
  file_path: string;
};

export const Route = createFileRoute("/market-data")({
  head: () => ({ meta: [{ title: "Market Data · Operator" }] }),
  component: MarketDataPage,
});

function MarketDataPage() {
  const qc = useQueryClient();
  const files = useQuery<MarketDataFile[]>({
    queryKey: ["market-data-files"],
    queryFn: async () => (await http.get("/api/v1/market-data/files")).data,
    retry: 0,
  });

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader title="Market Data" description="Crawl from MT5, import CSV, browse files" />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <CrawlForm onDone={() => qc.invalidateQueries({ queryKey: ["market-data-files"] })} />
        <ImportForm onDone={() => qc.invalidateQueries({ queryKey: ["market-data-files"] })} />
      </div>

      <PageSection
        title="Files"
        description="CSV files available on the backend"
        actions={
          <Button variant="outline" size="sm" onClick={() => files.refetch()}>
            Refresh
          </Button>
        }
      >
        {files.isLoading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : files.isError ? (
          <p className="text-sm text-destructive">{getErrorMessage(files.error)}</p>
        ) : !files.data?.length ? (
          <EmptyState
            icon={<FileSpreadsheet className="h-6 w-6" />}
            title="No CSV files yet"
            hint="Import a CSV or crawl from MT5 to populate this list."
          />
        ) : (
          <DataTable
            head={["Filename", "Size (MB)", "Path"]}
            rows={files.data.map((file) => [
              file.filename,
              file.size_mb?.toFixed?.(2) ?? file.size_mb,
              file.path,
            ])}
          />
        )}
      </PageSection>
    </div>
  );
}

function CrawlForm({ onDone }: { onDone: () => void }) {
  const [symbol, setSymbol] = useState("XAUUSD");
  const [timeframes, setTimeframes] = useState("H1,M5");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [result, setResult] = useState<CrawlTimeframeResult[] | null>(null);

  const mut = useMutation<CrawlResponse>({
    mutationFn: async () => {
      const payload = {
        symbol,
        timeframes: timeframes
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
        start_date: startDate,
        end_date: endDate,
      };
      // Set timeout to 0 (no timeout) because crawling M5 data for years can take a long time
      return (await http.post("/api/v1/market-data/crawl", payload, { timeout: 0 })).data;
    },
    onSuccess: (data) => {
      setResult(Array.isArray(data.results) ? data.results : []);
      toast.success("Crawl complete");
      onDone();
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });

  return (
    <PageSection title="Crawl Data" description="Pull bars from MT5 into the backend">
      <div className="grid grid-cols-2 gap-3">
        <Field label="Symbol">
          <Input value={symbol} onChange={(e) => setSymbol(e.target.value)} />
        </Field>
        <Field label="Timeframes (comma sep)">
          <Input value={timeframes} onChange={(e) => setTimeframes(e.target.value)} />
        </Field>
        <Field label="Start date">
          <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </Field>
        <Field label="End date">
          <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
        </Field>
      </div>
      <Button className="mt-4" onClick={() => mut.mutate()} disabled={mut.isPending}>
        <Download /> {mut.isPending ? "Crawling & Saving DB (May take minutes)..." : "Crawl"}
      </Button>
      {result && result.length > 0 && (
        <div className="mt-4">
          <DataTable
            head={["Timeframe", "Rows", "Status"]}
            rows={result.map((item) => [item.timeframe, item.rows, item.status])}
          />
        </div>
      )}
    </PageSection>
  );
}

function ImportForm({ onDone }: { onDone: () => void }) {
  const [filePath, setFilePath] = useState("");
  const [symbol, setSymbol] = useState("XAUUSD");
  const [timeframe, setTimeframe] = useState("H1");
  const [result, setResult] = useState<ImportCsvResult | null>(null);

  const mut = useMutation<ImportCsvResult>({
    mutationFn: async () =>
      (await http.post("/api/v1/market-data/import", { file_path: filePath, symbol, timeframe }))
        .data,
    onSuccess: (data) => {
      setResult(data);
      toast.success(`Imported ${data?.rows_imported ?? "?"} rows`);
      onDone();
    },
    onError: (e) => toast.error(getErrorMessage(e)),
  });

  return (
    <PageSection title="Import CSV" description="Load a CSV already on the backend filesystem">
      <div className="grid grid-cols-1 gap-3">
        <Field label="File path">
          <Input
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            placeholder="/data/xauusd_h1.csv"
          />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Symbol">
            <Input value={symbol} onChange={(e) => setSymbol(e.target.value)} />
          </Field>
          <Field label="Timeframe">
            <Input value={timeframe} onChange={(e) => setTimeframe(e.target.value)} />
          </Field>
        </div>
      </div>
      <Button className="mt-4" onClick={() => mut.mutate()} disabled={mut.isPending}>
        <Upload /> {mut.isPending ? "Importing…" : "Import"}
      </Button>
      {result && (
        <div className="mt-4 rounded-md border border-border bg-muted/40 p-3 font-mono text-xs">
          <div>
            rows_imported: <span className="text-success">{result.rows_imported}</span>
          </div>
          <div>symbol: {result.symbol}</div>
          <div>timeframe: {result.timeframe}</div>
          <div>file_path: {result.file_path}</div>
        </div>
      )}
    </PageSection>
  );
}

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs uppercase tracking-wider text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}

export function DataTable({
  head,
  rows,
}: {
  head: string[];
  rows: (string | number | React.ReactNode)[][];
}) {
  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/40 text-left text-xs uppercase tracking-wider text-muted-foreground">
          <tr>
            {head.map((h) => (
              <th key={h} className="px-3 py-2 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono text-xs">
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-border hover:bg-accent/40">
              {r.map((c, j) => (
                <td key={j} className="px-3 py-2">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  hint,
}: {
  icon?: React.ReactNode;
  title: string;
  hint: string;
}) {
  return (
    <div className="flex flex-col items-center gap-2 py-8 text-center">
      {icon && <div className="text-muted-foreground">{icon}</div>}
      <p className="text-sm font-medium text-foreground">{title}</p>
      <p className="max-w-md text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

export { StatusBadge };
