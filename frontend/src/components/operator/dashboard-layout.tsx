import { useState } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import { Activity, Database, Boxes, LineChart, CandlestickChart, Radio, Gauge, ChevronLeft, ChevronRight, Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { TopbarStatus } from "./topbar-status";

function DashboardBackground() {
  const tickerTop = Array(10).fill("[AUTO] XAUUSD: BUY 2340.50 (Conf: 96%) SL:2335 TP:2355 • ");
  const tickerBottom = Array(10).fill("[EXEC] ORDER #8492 FILLED • PROFIT SECURED +$450.00 • ");

  return (
    <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
      <style>{`
        @keyframes marquee {
          0% { transform: translateX(0%); }
          100% { transform: translateX(-50%); }
        }
        @keyframes floatUp {
          0% { transform: translateY(110vh) translateX(0); opacity: 0; }
          10% { opacity: 0.8; }
          90% { opacity: 0.8; }
          100% { transform: translateY(-10vh) translateX(20px); opacity: 0; }
        }
        @keyframes scanline {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
        .animate-marquee { animation: marquee 30s linear infinite; }
        .animate-marquee-slow { animation: marquee 45s linear infinite reverse; }
        .trade-float { animation: floatUp 8s ease-in-out infinite; }
      `}</style>

      {/* Base Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

      {/* Orbs */}
      <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] rounded-full bg-yellow-500/10 blur-[150px] animate-[pulse_8s_ease-in-out_infinite]" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] rounded-full bg-amber-600/10 blur-[150px] animate-[pulse_10s_ease-in-out_infinite_alternate]" />

      {/* Scanline */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-yellow-500/5 to-transparent h-[100px] w-full animate-[scanline_4s_linear_infinite]" />

      {/* Ticker Streams */}
      <div className="absolute top-[10%] left-0 w-[200vw] flex opacity-20 text-yellow-500 font-mono text-xs animate-marquee whitespace-nowrap">
        {tickerTop.map((text, i) => <span key={i} className="mx-4">{text}</span>)}
      </div>
      <div className="absolute bottom-[10%] left-0 w-[200vw] flex opacity-20 text-emerald-500 font-mono text-xs animate-marquee-slow whitespace-nowrap">
        {tickerBottom.map((text, i) => <span key={i} className="mx-4">{text}</span>)}
      </div>

      {/* Floating Trades */}
      <div className="absolute left-[15%] bottom-0 px-2 py-1 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded text-xs font-bold font-mono trade-float backdrop-blur-sm" style={{ animationDelay: "0s", animationDuration: "7s" }}>+ BUY XAUUSD</div>
      <div className="absolute left-[70%] bottom-0 px-2 py-1 bg-red-500/20 text-red-400 border border-red-500/30 rounded text-xs font-bold font-mono trade-float backdrop-blur-sm" style={{ animationDelay: "2s", animationDuration: "9s" }}>- SELL XAUUSD</div>
      <div className="absolute left-[40%] bottom-0 px-2 py-1 bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 rounded text-xs font-bold font-mono trade-float backdrop-blur-sm" style={{ animationDelay: "5s", animationDuration: "6s" }}>⚠ TRAIL STOP</div>
      <div className="absolute left-[85%] bottom-0 px-2 py-1 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded text-xs font-bold font-mono trade-float backdrop-blur-sm" style={{ animationDelay: "3.5s", animationDuration: "8s" }}>✓ TP HIT</div>
    </div>
  );
}

const NAV = [
  { to: "/dashboard", label: "Dashboard", icon: Gauge },
  { to: "/market-data", label: "Market Data", icon: Database },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/predictions", label: "Predictions", icon: LineChart },
  { to: "/trading", label: "Trading", icon: CandlestickChart },
  { to: "/monitor", label: "Monitor", icon: Radio },
] as const;

export function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-background text-foreground p-2 gap-2 md:p-4 md:gap-4 relative overflow-hidden animate-in fade-in zoom-in-95 duration-1000 ease-out fill-mode-both">
      <DashboardBackground />

      {/* Mobile Menu Overlay */}
      {isMobileMenuOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm md:hidden transition-opacity"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "shrink-0 flex-col rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-3xl shadow-[0_30px_60px_-15px_rgba(234,179,8,0.08)] transition-all duration-700 ease-out z-40 relative overflow-hidden",
          "absolute inset-y-2 left-2 md:relative md:inset-auto md:left-auto",
          isMobileMenuOpen ? "flex w-[240px]" : "hidden md:flex",
          isCollapsed && !isMobileMenuOpen ? "md:w-[80px]" : "md:w-[240px]"
        )}
      >
        {/* Subtle inner gradient to make it soft */}
        <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent opacity-50 pointer-events-none" />

        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="hidden md:flex absolute -right-3 top-6 h-6 w-6 items-center justify-center rounded-full border border-white/10 bg-slate-900 text-slate-400 hover:bg-yellow-500 hover:text-slate-950 hover:border-yellow-500 transition-colors z-20 shadow-lg"
        >
          {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>

        <button
          onClick={() => setIsMobileMenuOpen(false)}
          className="md:hidden absolute right-4 top-5 flex h-8 w-8 items-center justify-center rounded-lg bg-white/5 text-slate-400 hover:text-white"
        >
          <X className="h-5 w-5" />
        </button>

        <div className={cn("flex h-20 items-center gap-3 border-b border-white/5 transition-all duration-700 relative z-10", isCollapsed && !isMobileMenuOpen ? "justify-center px-0" : "px-6")}>
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-yellow-400/20 to-amber-600/10 border border-yellow-500/20 text-yellow-500 shrink-0 shadow-[0_0_20px_rgba(234,179,8,0.1)]">
            <Activity className="h-5 w-5" />
          </div>
          {(!isCollapsed || isMobileMenuOpen) && (
            <div className="flex flex-col leading-none overflow-hidden">
              <span className="font-display text-xl font-semibold tracking-tight whitespace-nowrap text-white">Operator</span>
              <span className="text-[11px] font-medium text-slate-400 whitespace-nowrap mt-1">
                Trading Console
              </span>
            </div>
          )}
        </div>
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1.5 overflow-x-hidden relative z-10">
          {NAV.map((item) => {
            const Icon = item.icon;
            const active = pathname.startsWith(item.to);
            return (
              <Link
                key={item.to}
                to={item.to}
                title={isCollapsed && !isMobileMenuOpen ? item.label : undefined}
                onClick={() => setIsMobileMenuOpen(false)}
                className={cn(
                  "flex items-center gap-3 rounded-2xl px-4 py-3 text-sm transition-all duration-500 ease-out relative overflow-hidden group font-medium",
                  active
                    ? "bg-white/[0.05] text-yellow-400 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)]"
                    : "text-slate-400 hover:bg-white/[0.03] hover:text-slate-200",
                  isCollapsed && !isMobileMenuOpen ? "justify-center" : "justify-start"
                )}
              >
                {active && <div className="absolute left-0 top-1/2 -translate-y-1/2 h-1/2 w-1 bg-gradient-to-b from-yellow-400 to-amber-600 rounded-r-full opacity-70" />}
                <Icon className={cn("shrink-0 transition-transform duration-500 ease-out group-hover:scale-105", isCollapsed && !isMobileMenuOpen ? "h-6 w-6" : "h-5 w-5")} />
                {(!isCollapsed || isMobileMenuOpen) && <span className="whitespace-nowrap">{item.label}</span>}
              </Link>
            );
          })}
        </nav>
        <div className={cn("border-t border-white/5 p-6 text-xs text-slate-500 font-medium transition-all duration-500 text-center relative z-10", isCollapsed && !isMobileMenuOpen ? "px-0" : "px-6")}>
          {(!isCollapsed || isMobileMenuOpen) ? "v1.0 · Connected" : "v1.0"}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-2xl shadow-lg overflow-hidden relative z-10">
        <header className="sticky top-0 z-20 flex h-20 items-center justify-between gap-4 border-b border-white/5 bg-[#0a0f16]/60 px-6 md:px-8 backdrop-blur-2xl">
          <div className="flex items-center gap-3 md:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(true)}
              className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/5 text-white hover:bg-white/10 transition-colors"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-yellow-400/20 to-amber-600/10 border border-yellow-500/20 text-yellow-500 hidden sm:flex shadow-[0_0_20px_rgba(234,179,8,0.1)]">
              <Activity className="h-5 w-5" />
            </div>
            <span className="font-display text-xl font-semibold tracking-tight hidden sm:inline text-white">Operator</span>
          </div>
          <div className="hidden md:block" />
          <TopbarStatus />
        </header>
        <main className="flex-1 overflow-y-auto px-4 py-6 md:px-8 md:py-8">{children}</main>
      </div>
    </div>
  );
}
