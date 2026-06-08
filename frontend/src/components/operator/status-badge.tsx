import { cn } from "@/lib/utils";

export type StatusTone = "success" | "warning" | "danger" | "neutral" | "info";

const TONE: Record<StatusTone, string> = {
  success: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  warning: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
  danger: "bg-red-500/10 text-red-500 border-red-500/20",
  neutral: "bg-white/5 text-slate-400 border-white/10",
  info: "bg-blue-500/10 text-blue-400 border-blue-500/20",
};

export function StatusBadge({
  tone = "neutral",
  children,
  className,
  dot = true,
}: {
  tone?: StatusTone;
  children: React.ReactNode;
  className?: string;
  dot?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-[11px] font-medium shadow-[inset_0_1px_0_0_rgba(255,255,255,0.05)]",
        TONE[tone],
        className,
      )}
    >
      {dot && (
        <span
          className={cn("h-1.5 w-1.5 rounded-full shadow-[0_0_8px_currentColor]", {
            "bg-emerald-400": tone === "success",
            "bg-yellow-500 animate-pulse": tone === "warning",
            "bg-red-500": tone === "danger",
            "bg-slate-400": tone === "neutral",
            "bg-blue-400": tone === "info",
          })}
        />
      )}
      {children}
    </span>
  );
}
