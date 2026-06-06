import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import {
  ArrowRight,
  Activity,
  ShieldCheck,
  CandlestickChart,
  TrendingUp,
  Coins,
} from "lucide-react";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/")({
  head: () => ({ meta: [{ title: "XAUUSD AI Predictor · Gold Trading Intelligence" }] }),
  component: LandingPage,
});

function LandingPage() {
  const [activeStep, setActiveStep] = useState<number | null>(null);

  const packetStyle =
    activeStep === 1
      ? { left: "-10%", width: "20%", opacity: 1, transition: "all 0.5s cubic-bezier(0.4, 0, 0.2, 1)" }
      : activeStep === 2
      ? { left: "40%", width: "20%", opacity: 1, transition: "all 0.5s cubic-bezier(0.4, 0, 0.2, 1)" }
      : activeStep === 3
      ? { left: "90%", width: "20%", opacity: 1, transition: "all 0.5s cubic-bezier(0.4, 0, 0.2, 1)" }
      : { left: "0%", width: "100%", opacity: 0.3, transition: "all 0.5s cubic-bezier(0.4, 0, 0.2, 1)" };

  const packetClass = "absolute top-0 h-full bg-gradient-to-r from-transparent via-yellow-400 to-transparent shadow-[0_0_15px_#eab308] rounded-full";

  return (
    <div className="min-h-screen bg-[#0a0f16] text-slate-50 selection:bg-yellow-500/30 font-sans overflow-x-hidden">
      <AutoTradingBackground />

      <div className="relative z-10">
        {/* Navbar */}
        <header className="sticky top-0 z-50 w-full border-b border-yellow-500/10 bg-[#0a0f16]/60 backdrop-blur-xl">
          <div className="container mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-yellow-400 to-amber-600 shadow-[0_0_20px_rgba(245,158,11,0.4)]">
                <CandlestickChart className="h-5 w-5 text-slate-950" />
              </div>
              <span className="text-xl font-bold tracking-tight text-white">
                GoldTrader <span className="text-yellow-500">AI</span>
              </span>
            </div>
            <nav className="hidden md:flex gap-8 text-sm font-medium text-slate-300">
              <a href="#features" className="hover:text-yellow-400 transition-colors">
                Features
              </a>
              <a href="#how-it-works" className="hover:text-yellow-400 transition-colors">
                How it Works
              </a>
            </nav>
            <div className="flex items-center gap-4">
              <Link to="/dashboard">
                <Button
                  variant="default"
                  className="bg-gradient-to-r from-yellow-400 to-amber-500 text-slate-950 hover:from-yellow-300 hover:to-amber-400 gap-2 h-9 px-5 rounded-full font-bold transition-all shadow-[0_0_20px_rgba(245,158,11,0.3)] border-0"
                >
                  Console <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 overflow-hidden">
          <div className="container mx-auto max-w-7xl px-6 text-center relative">
            {/* Animated Candlesticks Background Graphic */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-[300px] opacity-[0.03] pointer-events-none flex justify-center items-end gap-2 sm:gap-4 overflow-hidden">
              <div
                className="w-6 sm:w-12 bg-emerald-500 animate-[pulse_3s_ease-in-out_infinite]"
                style={{ height: "40%" }}
              ></div>
              <div
                className="w-6 sm:w-12 bg-red-500 animate-[pulse_4s_ease-in-out_infinite_alternate]"
                style={{ height: "20%" }}
              ></div>
              <div
                className="w-6 sm:w-12 bg-emerald-500 animate-[pulse_2.5s_ease-in-out_infinite]"
                style={{ height: "70%" }}
              ></div>
              <div
                className="w-6 sm:w-12 bg-emerald-500 animate-[pulse_3.5s_ease-in-out_infinite_alternate]"
                style={{ height: "90%" }}
              ></div>
              <div
                className="w-6 sm:w-12 bg-red-500 animate-[pulse_5s_ease-in-out_infinite]"
                style={{ height: "50%" }}
              ></div>
              <div
                className="w-6 sm:w-12 bg-emerald-500 animate-[pulse_4.5s_ease-in-out_infinite_alternate]"
                style={{ height: "80%" }}
              ></div>
            </div>

            <div className="inline-flex items-center gap-2 rounded-full border border-yellow-500/20 bg-yellow-500/10 px-4 py-1.5 text-sm font-medium text-yellow-400 mb-8 backdrop-blur-md animate-in fade-in slide-in-from-bottom-4 duration-700">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-yellow-500"></span>
              </span>
              XAUUSD Model v1.0 Active
            </div>

            <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8 text-transparent bg-clip-text bg-gradient-to-r from-white via-yellow-100 to-amber-500 animate-in fade-in slide-in-from-bottom-6 duration-700 delay-100">
              Master the Gold Market <br className="hidden md:block" /> with AI Precision
            </h1>

            <p className="max-w-2xl mx-auto text-lg md:text-xl text-slate-300 mb-10 leading-relaxed animate-in fade-in slide-in-from-bottom-6 duration-700 delay-200">
              Automate your XAUUSD trading strategy with high-precision Deep Learning models.
              Real-time multi-timeframe analysis executing directly on MetaTrader 5.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-300">
              <Link to="/dashboard">
                <Button
                  size="lg"
                  className="h-14 px-8 rounded-full bg-gradient-to-r from-yellow-500 to-amber-600 hover:from-yellow-400 hover:to-amber-500 text-slate-950 font-bold gap-2 shadow-[0_0_30px_rgba(245,158,11,0.5)] transition-all hover:scale-105 border-0"
                >
                  Launch Trading Console <TrendingUp className="h-5 w-5" />
                </Button>
              </Link>
              <a href="#features">
                <Button
                  size="lg"
                  variant="outline"
                  className="h-14 px-8 rounded-full border-yellow-500/20 bg-[#0a0f16]/50 text-yellow-400 hover:bg-yellow-500/10 hover:text-yellow-300 font-semibold transition-all backdrop-blur-md"
                >
                  Explore Features
                </Button>
              </a>
            </div>
          </div>
        </section>

        {/* Stats Section */}
        <section className="py-10 border-y border-yellow-500/10 bg-[#0a0f16]/80 backdrop-blur-lg">
          <div className="container mx-auto max-w-7xl px-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
              <div>
                <div className="text-3xl font-bold text-white mb-1">XAUUSD</div>
                <div className="text-sm text-slate-400 uppercase tracking-wider font-medium">
                  Primary Asset
                </div>
              </div>
              <div>
                <div className="text-3xl font-bold text-yellow-500 mb-1">H1 & M5</div>
                <div className="text-sm text-slate-400 uppercase tracking-wider font-medium">
                  Timeframes
                </div>
              </div>
              <div>
                <div className="text-3xl font-bold text-white mb-1">&lt; 50ms</div>
                <div className="text-sm text-slate-400 uppercase tracking-wider font-medium">
                  Execution Time
                </div>
              </div>
              <div>
                <div className="text-3xl font-bold text-emerald-400 mb-1">24/5</div>
                <div className="text-sm text-slate-400 uppercase tracking-wider font-medium">
                  Auto Trading
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section
          id="features"
          className="py-24 bg-gradient-to-b from-[#0a0f16]/60 to-[#05080b]/60 backdrop-blur-md"
        >
          <div className="container mx-auto max-w-7xl px-6">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">
                Engineered for Gold Trading
              </h2>
              <p className="text-slate-400 max-w-2xl mx-auto">
                Capitalize on XAUUSD volatility with features designed specifically for the precious
                metals market.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              <FeatureCard
                icon={<Activity className="h-6 w-6 text-yellow-500" />}
                title="Dual-Timeframe ML"
                description="Synchronized predictions across H1 (Trend) and M5 (Entry) timeframes to catch perfect gold swings."
              />
              <FeatureCard
                icon={<CandlestickChart className="h-6 w-6 text-amber-500" />}
                title="MT5 Native Integration"
                description="Direct bridge to MetaTrader 5 ensures your orders hit the market with zero latency."
              />
              <FeatureCard
                icon={<ShieldCheck className="h-6 w-6 text-yellow-400" />}
                title="Strict Risk Control"
                description="Built-in stop loss, trailing stops, and volume scaling to protect your capital during news events."
              />
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section
          id="how-it-works"
          className="py-24 relative overflow-hidden bg-[#05080b]/60 backdrop-blur-md border-y border-white/5"
        >
          <div className="container mx-auto max-w-7xl px-6 relative z-10">
            <div className="text-center mb-20">
              <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">
                Automate in 3 Steps
              </h2>
              <p className="text-slate-400 max-w-2xl mx-auto">
                From setup to live gold trading in minutes.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-12 relative">
              {/* Tech Connecting Line with animated data packet */}
              <div className="hidden md:block absolute top-[48px] left-[16%] right-[16%] h-[2px] bg-slate-800/50 overflow-hidden rounded-full">
                <div className={packetClass} style={packetStyle} />
              </div>

              <StepCard
                step="01"
                title="Connect MT5"
                description="Link your trading account. The console automatically syncs your XAUUSD market data."
                onMouseEnter={() => setActiveStep(1)}
                onMouseLeave={() => setActiveStep(null)}
              />
              <StepCard
                step="02"
                title="Initialize AI Models"
                description="Load the pre-trained Deep Learning models for H1 and M5 timeframes."
                onMouseEnter={() => setActiveStep(2)}
                onMouseLeave={() => setActiveStep(null)}
              />
              <StepCard
                step="03"
                title="Start Auto-Trade"
                description="Watch as the AI scans the charts, identifies patterns, and executes trades autonomously."
                onMouseEnter={() => setActiveStep(3)}
                onMouseLeave={() => setActiveStep(null)}
              />
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-white/5 bg-[#0a0f16]/80 backdrop-blur-lg py-12">
          <div className="container mx-auto max-w-7xl px-6 flex flex-col md:flex-row items-center justify-between">
            <div className="flex items-center gap-2 mb-4 md:mb-0">
              <Coins className="h-5 w-5 text-yellow-500" />
              <span className="font-bold text-white">GoldTrader AI</span>
            </div>
            <p className="text-sm text-slate-500">
              &copy; {new Date().getFullYear()} Operator Console. Built for XAUUSD.
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="group relative overflow-hidden rounded-2xl border border-white/5 bg-[#0a0f16]/60 p-8 transition-all hover:bg-[#0f172a]/80 hover:-translate-y-1 backdrop-blur-md">
      {/* Top glowing accent line */}
      <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-yellow-500/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

      <div className="absolute inset-0 bg-gradient-to-b from-yellow-500/5 to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100 pointer-events-none" />

      {/* Cyber-style icon container */}
      <div className="mb-6 relative inline-flex h-14 w-14 items-center justify-center rounded-xl bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 group-hover:text-yellow-300 group-hover:shadow-[0_0_20px_rgba(234,179,8,0.3)] transition-all duration-300">
        {icon}
        <div className="absolute inset-0 rounded-xl border border-yellow-400/0 group-hover:border-yellow-400/50 animate-[ping_2s_ease-out_infinite] opacity-0 group-hover:opacity-100" />
      </div>

      <h3 className="mb-3 text-xl font-bold text-white group-hover:text-yellow-400 transition-colors">
        {title}
      </h3>
      <p className="text-slate-400 leading-relaxed text-sm">{description}</p>
    </div>
  );
}

function StepCard({
  step,
  title,
  description,
  onMouseEnter,
  onMouseLeave,
}: {
  step: string;
  title: string;
  description: string;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}) {
  return (
    <div 
      className="relative z-10 flex flex-col items-center text-center group"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {/* Cyber Ring Container */}
      <div className="relative mb-8 flex h-24 w-24 items-center justify-center">
        {/* Outer rotating dashed ring */}
        <div className="absolute inset-0 rounded-full border-[2px] border-dashed border-slate-700 group-hover:border-yellow-500/50 transition-colors duration-500 group-hover:animate-[spin_4s_linear_infinite]" />

        {/* Inner solid glowing ring */}
        <div className="absolute inset-2 rounded-full border border-yellow-500/20 bg-yellow-500/5 group-hover:bg-yellow-500/20 transition-all duration-500" />

        {/* Number Core */}
        <div className="relative z-10 flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-yellow-400 to-amber-600 shadow-[0_0_20px_rgba(245,158,11,0.3)] group-hover:shadow-[0_0_30px_rgba(245,158,11,0.6)] transition-shadow">
          <span className="text-xl font-black text-slate-950 font-mono tracking-tighter">
            {step}
          </span>
        </div>
      </div>

      <h3 className="mb-3 text-xl font-bold text-white group-hover:text-yellow-400 transition-colors">
        {title}
      </h3>
      <p className="text-slate-400 text-sm">{description}</p>
    </div>
  );
}

function AutoTradingBackground() {
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
        .animate-marquee {
          animation: marquee 30s linear infinite;
        }
        .animate-marquee-slow {
          animation: marquee 45s linear infinite reverse;
        }
        .trade-float {
          animation: floatUp 8s ease-in-out infinite;
        }
      `}</style>

      {/* Base Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

      {/* Orbs */}
      <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] rounded-full bg-yellow-500/10 blur-[150px] animate-[pulse_8s_ease-in-out_infinite]" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] rounded-full bg-amber-600/10 blur-[150px] animate-[pulse_10s_ease-in-out_infinite_alternate]" />

      {/* Scanline */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-yellow-500/5 to-transparent h-[100px] w-full animate-[scanline_4s_linear_infinite]" />

      {/* Ticker Stream */}
      <div className="absolute top-[20%] left-0 w-[200vw] flex opacity-20 text-yellow-500 font-mono text-xs animate-marquee whitespace-nowrap">
        {tickerTop.map((text, i) => (
          <span key={i} className="mx-4">
            {text}
          </span>
        ))}
      </div>
      <div className="absolute top-[60%] left-0 w-[200vw] flex opacity-20 text-emerald-500 font-mono text-xs animate-marquee-slow whitespace-nowrap">
        {tickerBottom.map((text, i) => (
          <span key={i} className="mx-4">
            {text}
          </span>
        ))}
      </div>

      {/* Floating Trades */}
      <div
        className="absolute left-[15%] bottom-0 px-2 py-1 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded text-xs font-bold font-mono trade-float backdrop-blur-sm"
        style={{ animationDelay: "0s", animationDuration: "7s" }}
      >
        + BUY XAUUSD
      </div>
      <div
        className="absolute left-[70%] bottom-0 px-2 py-1 bg-red-500/20 text-red-400 border border-red-500/30 rounded text-xs font-bold font-mono trade-float backdrop-blur-sm"
        style={{ animationDelay: "2s", animationDuration: "9s" }}
      >
        - SELL XAUUSD
      </div>
      <div
        className="absolute left-[40%] bottom-0 px-2 py-1 bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 rounded text-xs font-bold font-mono trade-float backdrop-blur-sm"
        style={{ animationDelay: "5s", animationDuration: "6s" }}
      >
        ⚠ TRAIL STOP
      </div>
      <div
        className="absolute left-[85%] bottom-0 px-2 py-1 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded text-xs font-bold font-mono trade-float backdrop-blur-sm"
        style={{ animationDelay: "3.5s", animationDuration: "8s" }}
      >
        ✓ TP HIT
      </div>
      <div
        className="absolute left-[5%] bottom-0 px-2 py-1 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded text-xs font-bold font-mono trade-float backdrop-blur-sm"
        style={{ animationDelay: "7s", animationDuration: "10s" }}
      >
        + BUY XAUUSD
      </div>
      <div
        className="absolute left-[55%] bottom-0 px-2 py-1 bg-red-500/20 text-red-400 border border-red-500/30 rounded text-xs font-bold font-mono trade-float backdrop-blur-sm"
        style={{ animationDelay: "8s", animationDuration: "7.5s" }}
      >
        - SELL XAUUSD
      </div>
    </div>
  );
}
