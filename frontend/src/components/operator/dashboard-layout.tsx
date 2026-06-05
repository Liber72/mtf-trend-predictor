import { Link, useRouterState } from "@tanstack/react-router";
import { Activity, Database, Boxes, LineChart, CandlestickChart, Radio, Gauge } from "lucide-react";
import { cn } from "@/lib/utils";
import { TopbarStatus } from "./topbar-status";

const NAV = [
  { to: "/", label: "Dashboard", icon: Gauge },
  { to: "/market-data", label: "Market Data", icon: Database },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/predictions", label: "Predictions", icon: LineChart },
  { to: "/trading", label: "Trading", icon: CandlestickChart },
  { to: "/monitor", label: "Monitor", icon: Radio },
] as const;

export function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <aside className="hidden w-[220px] shrink-0 flex-col border-r border-sidebar-border bg-sidebar md:flex">
        <div className="flex h-14 items-center gap-2 border-b border-sidebar-border px-4">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/15 text-primary">
            <Activity className="h-4 w-4" />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-sm font-semibold tracking-tight">Operator</span>
            <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Trading Console
            </span>
          </div>
        </div>
        <nav className="flex-1 overflow-y-auto px-2 py-3">
          {NAV.map((item) => {
            const Icon = item.icon;
            const active = item.to === "/" ? pathname === "/" : pathname.startsWith(item.to);
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  "mb-0.5 flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors",
                  active
                    ? "bg-accent text-foreground"
                    : "text-sidebar-foreground hover:bg-accent/60 hover:text-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
                {active && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-primary" />}
              </Link>
            );
          })}
        </nav>
        <div className="border-t border-sidebar-border px-4 py-3 text-[10px] uppercase tracking-widest text-muted-foreground">
          v1 · internal
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-14 items-center justify-between gap-4 border-b border-border bg-background/80 px-4 backdrop-blur md:px-6">
          <div className="flex items-center gap-2 md:hidden">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/15 text-primary">
              <Activity className="h-4 w-4" />
            </div>
            <span className="text-sm font-semibold">Operator</span>
          </div>
          <TopbarStatus />
        </header>
        <main className="flex-1 px-4 py-6 md:px-6 md:py-8">{children}</main>
      </div>
    </div>
  );
}
