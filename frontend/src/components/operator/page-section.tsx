import { cn } from "@/lib/utils";

export function PageSection({
  title,
  description,
  actions,
  children,
  className,
}: {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-3xl shadow-xl overflow-hidden transition-all duration-700 ease-out hover:bg-white/[0.03]", className)}>
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/5 px-6 py-5 bg-white/[0.01]">
        <div className="min-w-0">
          <h2 className="text-[15px] font-semibold text-slate-200 tracking-wide">{title}</h2>
          {description && <p className="mt-1 text-[13px] text-slate-400 font-medium">{description}</p>}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-3">{actions}</div>}
      </header>
      <div className="p-6">{children}</div>
    </section>
  );
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-10 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
      <div>
        <h1 className="font-display text-4xl font-semibold tracking-tight text-white">{title}</h1>
        {description && <p className="mt-3 text-[15px] text-slate-400 font-medium">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
