import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from "react";
import ChatPanel from "./components/ChatPanel";
import { getDashboardMetrics, login } from "./lib/api";
import { useGeoStore } from "./store/useGeoStore";
import type { AuthUser, DashboardMetrics, SelectedAreaPayload } from "./types";

const Map3D = lazy(() => import("./components/Map3D"));

type AppTab = "workspace" | "dashboard";

const TOKEN_KEY = "aurbital_auth_token";
const USER_KEY = "aurbital_auth_user";

export default function App() {
  const { selectedArea, setSelectedArea } = useGeoStore();
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [activeTab, setActiveTab] = useState<AppTab>("workspace");

  const [username, setUsername] = useState("researcher");
  const [password, setPassword] = useState("earthaware123");
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  useEffect(() => {
    const savedToken = localStorage.getItem(TOKEN_KEY);
    const savedUser = localStorage.getItem(USER_KEY);
    if (savedToken) setToken(savedToken);
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser) as AuthUser);
      } catch {
        localStorage.removeItem(USER_KEY);
      }
    }
  }, []);

  const onAreaSelected = useCallback(
    (area: SelectedAreaPayload) => {
      setSelectedArea(area);
    },
    [setSelectedArea]
  );

  const isAuthed = useMemo(() => Boolean(token && user), [token, user]);

  const onLogin = async () => {
    setLoginLoading(true);
    setLoginError(null);
    try {
      const res = await login(username.trim(), password);
      setToken(res.token);
      setUser(res.user);
      localStorage.setItem(TOKEN_KEY, res.token);
      localStorage.setItem(USER_KEY, JSON.stringify(res.user));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setLoginError(msg);
    } finally {
      setLoginLoading(false);
    }
  };

  const onLogout = () => {
    setToken(null);
    setUser(null);
    setMetrics(null);
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  };

  useEffect(() => {
    if (!token || activeTab !== "dashboard") return;
    let mounted = true;
    setMetricsLoading(true);
    setMetricsError(null);

    getDashboardMetrics(token)
      .then((data) => {
        if (!mounted) return;
        setMetrics(data);
      })
      .catch((err) => {
        if (!mounted) return;
        const msg = err instanceof Error ? err.message : String(err);
        setMetricsError(msg);
      })
      .finally(() => {
        if (!mounted) return;
        setMetricsLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [token, activeTab]);

  if (!isAuthed) {
    return (
      <div className="relative flex h-full items-center justify-center bg-[#050505] p-4 overflow-hidden">
        {/* Background Decorative Elements */}
        <div className="absolute -top-24 -left-24 h-96 w-96 rounded-full bg-cyan-500/5 blur-[120px]"></div>
        <div className="absolute -bottom-24 -right-24 h-96 w-96 rounded-full bg-blue-500/5 blur-[120px]"></div>
        
        <div className="w-full max-w-md relative">
          <div className="absolute inset-0 rounded-2xl bg-gradient-to-b from-cyan-400/20 to-transparent blur-sm -m-1"></div>
          
          <div className="relative rounded-2xl border border-cyan-400/20 bg-[#0a0a0a]/80 backdrop-blur-xl p-8 shadow-2xl">
            <div className="mb-8 text-center">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-cyan-400/30 bg-cyan-500/10 text-3xl shadow-[0_0_20px_rgba(34,211,238,0.2)]">
                🛰️
              </div>
              <h1 className="text-2xl font-bold tracking-tight text-white">
                TERRA<span className="text-cyan-400">SIGHT</span>
              </h1>
              <p className="mt-2 text-sm text-slate-400">Multimodal Geo-Intelligence Research Platform</p>
            </div>

            <div className="space-y-4">
              <div>
                <label className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-slate-500">Authorized User</label>
                <div className="relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">👤</span>
                  <input
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="Enter researcher ID"
                    className="w-full rounded-lg border border-white/10 bg-white/5 py-2.5 pl-10 pr-4 text-sm text-white placeholder:text-slate-600 outline-none focus:border-cyan-400/50 focus:bg-white/10 transition-all"
                  />
                </div>
              </div>

              <div>
                <label className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-slate-500">Access Key</label>
                <div className="relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">🔑</span>
                  <input
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    type="password"
                    placeholder="••••••••"
                    className="w-full rounded-lg border border-white/10 bg-white/5 py-2.5 pl-10 pr-4 text-sm text-white placeholder:text-slate-600 outline-none focus:border-cyan-400/50 focus:bg-white/10 transition-all"
                    onKeyDown={(e) => {
                      if (e.key === "Enter") void onLogin();
                    }}
                  />
                </div>
              </div>

              <button
                onClick={() => void onLogin()}
                disabled={loginLoading}
                className="group relative mt-2 w-full overflow-hidden rounded-lg bg-cyan-500 py-3 text-sm font-semibold text-black transition-all hover:bg-cyan-400 active:scale-[0.98] disabled:opacity-50"
              >
                <div className="absolute inset-0 flex items-center justify-center bg-white/20 opacity-0 transition-opacity group-hover:opacity-100"></div>
                {loginLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="h-4 w-4 animate-spin text-black" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    AUTHENTICATING...
                  </span>
                ) : (
                  "ACCESS COMMAND CENTER"
                )}
              </button>
            </div>

            {loginError && (
              <div className="mt-6 flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-xs text-red-200 animate-in fade-in slide-in-from-top-2">
                <span>⚠️</span>
                <span>{loginError}</span>
              </div>
            )}

            <div className="mt-8 border-t border-white/5 pt-6 text-center">
              <p className="text-[10px] uppercase tracking-[0.2em] text-slate-600">
                Secure Terminal — ISRO Multispectral Access
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col bg-[#050505] font-['Inter'] text-[#c8ccd0]">
      {/* War Room Header Bar */}
      <div className="flex items-center justify-between border-b border-[#1a1f2e] bg-gradient-to-r from-[#0a0a0a] via-[#111318] to-[#0a0a0a] px-5 py-2 header-font text-[0.8rem]">
        <div className="flex items-center gap-3">
          <span className="text-[1.3rem] font-bold tracking-[2px] text-[#00e5ff] header-font">AURBITAL</span>
          <span className="text-[0.65rem] text-[#4a5568] data-font">v2.4.0 CORE_INTEL</span>
        </div>
        
        <div className="hidden items-center gap-5 md:flex">
          <div className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-[#00e676] shadow-[0_0_6px_#00e676]"></span>
            <span className="text-xs">MODEL_READY</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-[#00e676] shadow-[0_0_6px_#00e676]"></span>
            <span className="text-xs">GEO_STREAMS: ACTIVE</span>
          </div>
          <div className="text-[0.7rem] text-[#4a5568]">
            {new Date().toISOString().replace('T', ' ').substring(0, 19)} UTC
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setActiveTab("workspace")}
            className={`rounded px-3 py-1 text-xs transition-colors ${activeTab === "workspace" ? "bg-cyan-500/20 text-[#00e5ff] border border-cyan-500/40" : "text-[#4a5568] hover:text-slate-300"}`}
          >
            WORKSPACE
          </button>
          <button
            onClick={() => setActiveTab("dashboard")}
            className={`rounded px-3 py-1 text-xs transition-colors ${activeTab === "dashboard" ? "bg-cyan-500/20 text-[#00e5ff] border border-cyan-500/40" : "text-[#4a5568] hover:text-slate-300"}`}
          >
            DASHBOARD
          </button>
          <div className="h-4 w-px bg-[#1a1f2e]"></div>
          <button onClick={onLogout} className="text-xs text-red-400/70 hover:text-red-400">
            [LOGOUT]
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-hidden p-3 md:p-4">
        {activeTab === "workspace" ? (
          <div className="grid h-full grid-cols-1 gap-4 lg:grid-cols-5">
            <div className="relative flex flex-col overflow-hidden rounded border border-[#1a1f2e] bg-[#0a0a0a] lg:col-span-3">
              <Suspense fallback={<div className="flex h-full items-center justify-center text-[#4a5568] font-['JetBrains_Mono'] text-xs">INITIALIZING_MAP_ENGINE...</div>}>
                <Map3D onAreaSelected={onAreaSelected} />
              </Suspense>
              
              {/* Map Overlay Info */}
              <div className="absolute bottom-4 left-4 pointer-events-none rounded border border-[#1a1f2e] bg-[#0a0e14]/80 p-2 font-['JetBrains_Mono'] text-[0.65rem] backdrop-blur-sm">
                <div className="text-[#00e5ff]">SENSOR: SENTINEL-2_MSI</div>
                <div className="text-[#4a5568]">LAT: {selectedArea?.center.lat.toFixed(4) || "0.0000"}</div>
                <div className="text-[#4a5568]">LNG: {selectedArea?.center.lng.toFixed(4) || "0.0000"}</div>
              </div>
            </div>
            
            <div className="lg:col-span-2 overflow-hidden h-full">
              <ChatPanel />
            </div>
          </div>
        ) : (
          <div className="h-full overflow-y-auto rounded border border-[#1a1f2e] bg-[#0a0e14] p-5 font-['JetBrains_Mono']">
            <div className="mb-6 flex items-center justify-between border-b border-[#1a1f2e] pb-4">
              <div>
                <h2 className="text-[1rem] font-bold tracking-wider text-[#00e5ff]">INTEL_DASHBOARD</h2>
                <p className="mt-1 text-[0.7rem] text-[#4a5568]">Cross-referencing training checkpoints with real-time RL feedback.</p>
              </div>
              <div className="rounded bg-[#00e676] px-2 py-0.5 text-[0.6rem] font-bold text-black uppercase">Live System</div>
            </div>

            {metricsLoading && <div className="text-xs text-[#4a5568] animate-pulse">FETCHING_REMOTE_METRICS...</div>}
            {metricsError && <div className="rounded border border-red-900/50 bg-red-950/20 p-3 text-xs text-red-400">ERROR: {metricsError}</div>}

            {metrics && (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                <TechnicalMetricCard label="Best Validation Loss" value={metrics.training.bestValLoss ?? "N/A"} status="green" />
                <TechnicalMetricCard label="Total Epochs" value={metrics.training.epochs ?? "N/A"} status="amber" />
                <TechnicalMetricCard label="Global Steps" value={metrics.training.globalSteps ?? "N/A"} status="blue" />
                <TechnicalMetricCard label="RL Feedback Log" value={metrics.rl.feedbackCount} status="green" />
              </div>
            )}

            {metrics && (
              <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-2">
                <div className="rounded border border-[#1a1f2e] bg-[#0d1117] p-4">
                  <h3 className="mb-3 text-xs font-bold uppercase tracking-widest text-[#ff9100]">Recent Performance</h3>
                  <div className="space-y-2">
                    <MetricLine label="Last Epoch" value={metrics.training.lastEpoch?.epoch ?? "N/A"} />
                    <MetricLine label="Train Loss" value={metrics.training.lastEpoch?.trainLoss ?? "N/A"} />
                    <MetricLine label="Val Loss" value={metrics.training.lastEpoch?.valLoss ?? "N/A"} />
                    <MetricLine label="Learning Rate" value={metrics.training.lastEpoch?.lr ?? "N/A"} />
                  </div>
                </div>
                
                <div className="rounded border border-[#1a1f2e] bg-[#0d1117] p-4">
                  <h3 className="mb-3 text-xs font-bold uppercase tracking-widest text-[#ff9100]">System Logs (Raw JSON)</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <button onClick={() => console.log(metrics.evaluation.day5)} className="rounded border border-[#1a1f2e] bg-[#0a0a0a] py-2 text-[0.6rem] text-[#00e5ff] hover:bg-[#1a1f2e]">EXAMINE_DAY5_JSON</button>
                    <button onClick={() => console.log(metrics.evaluation.baseline)} className="rounded border border-[#1a1f2e] bg-[#0a0a0a] py-2 text-[0.6rem] text-[#00e5ff] hover:bg-[#1a1f2e]">EXAMINE_BASELINE_JSON</button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function TechnicalMetricCard({ label, value, status }: { label: string; value: string | number; status: 'green' | 'amber' | 'red' | 'blue' }) {
  const statusColor = {
    green: 'text-[#00e676]',
    amber: 'text-[#ff9100]',
    red: 'text-[#ff1744]',
    blue: 'text-[#00e5ff]'
  }[status];

  return (
    <div className="rounded border border-[#1a1f2e] bg-[#0d1117] p-4 font-['JetBrains_Mono']">
      <div className="text-[0.65rem] uppercase tracking-widest text-[#4a5568]">{label}</div>
      <div className={`mt-2 text-[1.1rem] font-bold ${statusColor}`}>{String(value)}</div>
    </div>
  );
}

function MetricLine({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between border-b border-[#1a1f2e]/50 py-1 text-[0.7rem]">
      <span className="text-[#4a5568]">{label}:</span>
      <span className="font-bold text-[#e0e0e0]">{String(value)}</span>
    </div>
  );
}

