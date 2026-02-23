'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useSession, signOut } from 'next-auth/react';
import { useRouter } from 'next/navigation';

interface Analysis {
  id: string;
  tic_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  result?: {
    detection: boolean;
    confidence: number;
    period_days: number | null;
    vetting: { disposition: string };
  };
}

interface StatsData {
  objectsProcessed: number;
  detections: number;
  modelAccuracy: number;
  vettedCandidates: number;
  lastCalibration: string | null;
  driftDetected: boolean;
}

interface LiveTOI {
  tic_id: string;
  toi: string;
  period_days: number;
  depth_ppm: number;
  duration_hours: number;
  disposition: string;
  planet_radius_earth: number | null;
  star_tmag: number | null;
}

export default function DashboardPage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [stats, setStats] = useState<StatsData>({
    objectsProcessed: 0,
    detections: 0,
    modelAccuracy: 98.0,
    vettedCandidates: 0,
    lastCalibration: null,
    driftDetected: false,
  });
  const [liveTOI, setLiveTOI] = useState<LiveTOI[]>([]);
  const [toiLoading, setToiLoading] = useState(true);
  const [showAllLive, setShowAllLive] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showUserMenu, setShowUserMenu] = useState(false);

  useEffect(() => {
    if (status === 'unauthenticated') router.push('/auth/login?callbackUrl=/dashboard');
  }, [status, router]);

  const getUserInitial = () => {
    if (session?.user?.name) return session.user.name.charAt(0).toUpperCase();
    if (session?.user?.email) return session.user.email.charAt(0).toUpperCase();
    return '?';
  };

  const formatTimeSince = (iso: string | null): string => {
    if (!iso) return 'Never';
    const diff = Date.now() - new Date(iso).getTime();
    const m = Math.floor(diff / 60000);
    if (m < 1) return 'Just now';
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
  };

  const fetchData = async () => {
    try {
      setError(null);
      const [analysesRes, calibRes] = await Promise.all([
        fetch('/api/v1/analyses'),
        fetch('/api/v1/calibration/status'),
      ]);
      if (analysesRes.status === 401) { router.push('/auth/login?callbackUrl=/dashboard'); return; }

      let modelAccuracy = 98.0, lastCalibration: string | null = null, driftDetected = false;
      if (calibRes.ok) {
        const c = await calibRes.json();
        modelAccuracy  = c.accuracy       ?? 98.0;
        lastCalibration = c.last_calibration ?? null;
        driftDetected  = c.drift_detected  ?? false;
      }

      if (analysesRes.ok) {
        const d = await analysesRes.json();
        const list: Analysis[] = d.analyses || [];
        setAnalyses(list);
        const done = list.filter(a => a.status === 'completed');
        setStats({
          objectsProcessed: done.length,
          detections: done.filter(a => a.result?.detection).length,
          modelAccuracy,
          vettedCandidates: done.filter(a => a.result?.vetting?.disposition === 'PLANET_CANDIDATE').length,
          lastCalibration,
          driftDetected,
        });
      } else {
        setError('Failed to load dashboard data.');
      }
    } catch {
      setError('Unable to connect to server.');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchLiveTOI = async () => {
    try {
      const res = await fetch('/api/v1/toi-candidates');
      if (res.ok) {
        const d = await res.json();
        if (d.candidates) setLiveTOI(d.candidates);
      }
    } catch { /* silent */ } finally {
      setToiLoading(false);
    }
  };

  useEffect(() => { fetchData(); fetchLiveTOI(); }, []);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    setToiLoading(true);
    await Promise.all([fetchData(), fetchLiveTOI()]);
    setIsRefreshing(false);
  };

  const dispositionStyle: Record<string, string> = {
    CP:  'bg-[#e6f4ea] text-[#137333]',
    KP:  'bg-[#e6f4ea] text-[#137333]',
    PC:  'bg-[#fef7e0] text-[#b06000]',
    APC: 'bg-[#fef7e0] text-[#b06000]',
    FP:  'bg-[#fce8e6] text-[#c5221f]',
  };

  if (status === 'loading' || isLoading) {
    return (
      <div className="min-h-screen bg-[#f1f3f4] flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-[#1a73e8] border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-[#5f6368]">Loading dashboard…</p>
        </div>
      </div>
    );
  }
  if (status === 'unauthenticated') return null;

  return (
    <div className="min-h-screen bg-white">

      {/* ── Header ── */}
      <header className="fixed top-0 left-0 right-0 h-16 bg-white border-b border-[#dadce0] z-50 flex items-center px-4 gap-3">
        <button
          onClick={() => setSidebarOpen(v => !v)}
          className="w-10 h-10 hover:bg-[#f1f3f4] rounded-full flex items-center justify-center"
        >
          <svg className="w-5 h-5 text-[#5f6368]" fill="currentColor" viewBox="0 0 24 24">
            <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z" />
          </svg>
        </button>
        <Link href="/" className="text-[20px] font-medium text-[#202124] ml-1">
          Larun<span className="text-[#1a73e8]">.</span><span className="text-[#1a73e8]">Space</span>
        </Link>

        <div className="flex-1" />

        {/* User menu */}
        <div className="relative">
          <button
            onClick={() => setShowUserMenu(v => !v)}
            className="w-9 h-9 bg-[#1a73e8] rounded-full flex items-center justify-center text-white text-sm font-medium overflow-hidden"
          >
            {session?.user?.image
              ? <img src={session.user.image} alt="" className="w-9 h-9 rounded-full" />
              : getUserInitial()}
          </button>
          {showUserMenu && (
            <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-lg border border-[#dadce0] py-2 z-50">
              <div className="px-4 py-3 border-b border-[#dadce0]">
                <p className="text-sm font-medium text-[#202124]">{session?.user?.name || 'User'}</p>
                <p className="text-xs text-[#5f6368]">{session?.user?.email}</p>
              </div>
              <Link href="/cloud/billing" className="block px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4]" onClick={() => setShowUserMenu(false)}>
                Usage &amp; Billing
              </Link>
              <button
                onClick={() => signOut({ callbackUrl: '/' })}
                className="w-full text-left px-4 py-2 text-sm text-[#c5221f] hover:bg-[#fce8e6]"
              >
                Sign out
              </button>
            </div>
          )}
        </div>
      </header>

      {/* ── Sidebar ── */}
      <aside className={`fixed top-16 left-0 bottom-0 w-[240px] bg-white border-r border-[#dadce0] overflow-y-auto transition-transform z-40 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <nav className="py-3 px-3 space-y-0.5">

          <SideLink href="/dashboard" icon="dashboard" label="Dashboard" active />

          <p className="text-[10px] font-semibold text-[#9aa0a6] uppercase tracking-wider px-4 pt-5 pb-1">Workspace</p>
          <SideLink href="/cloud/analyze" icon="analyze" label="Run Analysis" />
          <SideLink href="/models"        icon="model"   label="Models" />

          <p className="text-[10px] font-semibold text-[#9aa0a6] uppercase tracking-wider px-4 pt-5 pb-1">Account</p>
          <SideLink href="/cloud/billing" icon="billing" label="Usage &amp; Billing" />

          <p className="text-[10px] font-semibold text-[#9aa0a6] uppercase tracking-wider px-4 pt-5 pb-1">Help</p>
          <SideLink href="/guide" icon="docs"   label="Documentation" />
          <SideLink href="/faq"   icon="faq"    label="FAQ" />
        </nav>
      </aside>

      {/* ── Main ── */}
      <main className={`pt-16 min-h-screen bg-[#f1f3f4] transition-all ${sidebarOpen ? 'lg:ml-[240px]' : ''}`}>
        <div className="p-6 max-w-7xl mx-auto space-y-6">

          {/* Error */}
          {error && (
            <div className="bg-[#fce8e6] border border-[#f5c6cb] rounded-lg p-4 flex items-center justify-between">
              <span className="text-sm text-[#c5221f]">{error}</span>
              <button onClick={() => { setError(null); fetchData(); }}
                className="px-3 py-1.5 text-xs bg-[#c5221f] text-white rounded">Retry</button>
            </div>
          )}

          {/* Page title + actions */}
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-normal text-[#202124]">Dashboard</h1>
            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="flex items-center gap-2 px-4 py-2 text-sm bg-white border border-[#dadce0] text-[#3c4043] rounded-lg hover:bg-[#f1f3f4] transition-colors disabled:opacity-50"
            >
              {isRefreshing && (
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              )}
              Refresh
            </button>
          </div>

          {/* ── Stats ── */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard label="Analyses Run" value={stats.objectsProcessed} sub={stats.objectsProcessed > 0 ? 'completed' : 'Run your first analysis'} />
            <StatCard label="Detections" value={stats.detections} sub={stats.detections > 0 ? 'transit signals found' : 'No detections yet'} />
            <StatCard
              label="Model Accuracy"
              value={`${stats.modelAccuracy}%`}
              sub={stats.driftDetected ? 'Drift detected' : `Calibrated ${formatTimeSince(stats.lastCalibration)}`}
              warn={stats.driftDetected}
            />
            <StatCard label="Planet Candidates" value={stats.vettedCandidates} sub={stats.vettedCandidates > 0 ? 'from vetting' : 'None yet'} />
          </div>

          {/* ── My Analyses ── */}
          <div className="bg-white rounded-lg shadow-[0_1px_2px_0_rgba(60,64,67,0.3)] overflow-hidden">
            <div className="px-6 py-4 border-b border-[#f1f3f4] flex items-center justify-between">
              <h2 className="text-base font-medium text-[#202124]">My Analyses</h2>
              <Link href="/cloud/analyze"
                className="text-sm text-[#1a73e8] hover:underline font-medium">
                + New Analysis
              </Link>
            </div>

            {analyses.length === 0 ? (
              <div className="px-6 py-12 text-center">
                <svg className="w-12 h-12 text-[#dadce0] mx-auto mb-3" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z" />
                </svg>
                <p className="text-sm text-[#5f6368] mb-4">No analyses yet. Start by running one on a TESS target.</p>
                <Link href="/cloud/analyze"
                  className="inline-block px-5 py-2 bg-[#1a73e8] text-white text-sm font-medium rounded-lg hover:bg-[#1557b0] transition-colors">
                  Run Analysis →
                </Link>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-[13px]">
                  <thead>
                    <tr className="text-left text-[#5f6368] border-b border-[#f1f3f4]">
                      <th className="py-3 px-6 font-medium">TIC ID</th>
                      <th className="py-3 px-4 font-medium">Status</th>
                      <th className="py-3 px-4 font-medium">Confidence</th>
                      <th className="py-3 px-4 font-medium">Period</th>
                      <th className="py-3 px-4 font-medium">Disposition</th>
                      <th className="py-3 px-4 font-medium">Date</th>
                      <th className="py-3 px-4 font-medium" />
                    </tr>
                  </thead>
                  <tbody>
                    {analyses.slice(0, 20).map(a => {
                      const statusStyle: Record<string, string> = {
                        completed:  'bg-[#e6f4ea] text-[#137333]',
                        processing: 'bg-[#e8f0fe] text-[#1a73e8]',
                        pending:    'bg-[#fef7e0] text-[#b06000]',
                        failed:     'bg-[#fce8e6] text-[#c5221f]',
                      };
                      return (
                        <tr key={a.id} className="border-b border-[#f1f3f4] hover:bg-[#f8f9fa]">
                          <td className="py-3 px-6 font-mono text-[#202124] font-medium">TIC {a.tic_id}</td>
                          <td className="py-3 px-4">
                            <span className={`px-2 py-0.5 rounded text-xs font-medium capitalize ${statusStyle[a.status]}`}>
                              {a.status}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-[#3c4043]">
                            {a.result?.confidence != null
                              ? `${(a.result.confidence * 100).toFixed(1)}%`
                              : '—'}
                          </td>
                          <td className="py-3 px-4 text-[#3c4043]">
                            {a.result?.period_days != null
                              ? `${a.result.period_days.toFixed(3)} d`
                              : '—'}
                          </td>
                          <td className="py-3 px-4">
                            {a.result?.vetting?.disposition
                              ? <span className="text-xs text-[#5f6368]">{a.result.vetting.disposition.replace(/_/g, ' ')}</span>
                              : '—'}
                          </td>
                          <td className="py-3 px-4 text-[#5f6368]">
                            {formatTimeSince(a.created_at)}
                          </td>
                          <td className="py-3 px-4 text-right">
                            {a.status === 'completed' && (
                              <Link href={`/results/${a.id}`}
                                className="text-xs text-[#1a73e8] hover:underline font-medium">
                                View →
                              </Link>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                {analyses.length > 20 && (
                  <p className="text-xs text-[#5f6368] px-6 py-3">
                    Showing 20 of {analyses.length} analyses.
                  </p>
                )}
              </div>
            )}
          </div>

          {/* ── TOI Candidates — only show unanalyzed targets ── */}
          {(() => {
            const analyzedTics = new Set(analyses.map(a => a.tic_id));
            const unanalyzed = liveTOI.filter(t => !analyzedTics.has(t.tic_id));
            return (
              <div className="bg-white rounded-lg shadow-[0_1px_2px_0_rgba(60,64,67,0.3)] overflow-hidden">
                <div className="px-6 py-4 border-b border-[#f1f3f4] flex items-center gap-3">
                  <h2 className="text-base font-medium text-[#202124]">TESS Candidates</h2>
                  {toiLoading
                    ? <span className="w-3.5 h-3.5 border-2 border-[#1a73e8] border-t-transparent rounded-full animate-spin" />
                    : unanalyzed.length > 0 && (
                      <span className="text-xs px-2 py-0.5 bg-[#e8f0fe] text-[#1a73e8] rounded-full font-medium">
                        {unanalyzed.length} unanalyzed
                      </span>
                    )
                  }
                </div>

                {!toiLoading && liveTOI.length === 0 ? (
                  <div className="px-6 py-8 text-center text-sm text-[#5f6368]">
                    Could not load live candidates from NASA. Try refreshing.
                  </div>
                ) : !toiLoading && unanalyzed.length === 0 ? (
                  <div className="px-6 py-10 text-center">
                    <svg className="w-10 h-10 text-[#34a853] mx-auto mb-3" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                    </svg>
                    <p className="text-sm font-medium text-[#202124] mb-1">All candidates analyzed</p>
                    <p className="text-xs text-[#5f6368]">Every loaded NASA target has been processed. View results in My Analyses above.</p>
                  </div>
                ) : (
                  <>
                    <div className="overflow-x-auto">
                      <table className="w-full text-[13px]">
                        <thead>
                          <tr className="text-left text-[#5f6368] border-b border-[#f1f3f4]">
                            <th className="py-3 px-6 font-medium">TOI</th>
                            <th className="py-3 px-4 font-medium">TIC ID</th>
                            <th className="py-3 px-4 font-medium">Period (d)</th>
                            <th className="py-3 px-4 font-medium">Depth (ppm)</th>
                            <th className="py-3 px-4 font-medium">Rp (R⊕)</th>
                            <th className="py-3 px-4 font-medium">Tmag</th>
                            <th className="py-3 px-4 font-medium">Disp.</th>
                            <th className="py-3 px-4 font-medium" />
                          </tr>
                        </thead>
                        <tbody>
                          {(showAllLive ? unanalyzed : unanalyzed.slice(0, 20)).map((toi, i) => (
                            <tr key={i} className="border-b border-[#f1f3f4] hover:bg-[#f8f9fa]">
                              <td className="py-3 px-6 font-medium text-[#202124]">TOI-{toi.toi}</td>
                              <td className="py-3 px-4 font-mono text-xs text-[#3c4043]">{toi.tic_id}</td>
                              <td className="py-3 px-4 text-[#3c4043]">{toi.period_days.toFixed(3)}</td>
                              <td className="py-3 px-4 text-[#3c4043]">{Math.round(toi.depth_ppm).toLocaleString()}</td>
                              <td className="py-3 px-4 text-[#3c4043]">{toi.planet_radius_earth?.toFixed(2) ?? '—'}</td>
                              <td className="py-3 px-4 text-[#3c4043]">{toi.star_tmag?.toFixed(1) ?? '—'}</td>
                              <td className="py-3 px-4">
                                <span className={`px-2 py-0.5 text-xs rounded font-medium ${dispositionStyle[toi.disposition] ?? 'bg-[#f1f3f4] text-[#5f6368]'}`}>
                                  {toi.disposition}
                                </span>
                              </td>
                              <td className="py-3 px-4 text-right">
                                <Link href={`/cloud/analyze?tic=${toi.tic_id}`}
                                  className="text-xs bg-[#1a73e8] hover:bg-[#1557b0] text-white px-3 py-1 rounded font-medium transition-colors">
                                  Analyze →
                                </Link>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {unanalyzed.length > 20 && (
                      <div className="px-6 py-3 border-t border-[#f1f3f4]">
                        <button onClick={() => setShowAllLive(v => !v)}
                          className="text-xs text-[#1a73e8] hover:underline font-medium">
                          {showAllLive ? 'Show fewer' : `Show all ${unanalyzed.length} candidates →`}
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          })()}

          {/* Footer */}
          <footer className="text-center py-6 text-xs text-[#9aa0a6] space-x-4">
            <Link href="/guide" className="hover:text-[#5f6368]">Docs</Link>
            <Link href="/faq"   className="hover:text-[#5f6368]">FAQ</Link>
            <Link href="/cloud/billing" className="hover:text-[#5f6368]">Billing</Link>
            <span>&copy; {new Date().getFullYear()} Larun.Space</span>
          </footer>

        </div>
      </main>
    </div>
  );
}


// ── Small reusable components ──────────────────────────────────────

function StatCard({ label, value, sub, warn }: { label: string; value: string | number; sub: string; warn?: boolean }) {
  return (
    <div className="bg-white rounded-lg p-5 shadow-[0_1px_2px_0_rgba(60,64,67,0.3)]">
      <p className="text-xs font-medium text-[#5f6368] uppercase tracking-wider mb-2">{label}</p>
      <p className={`text-3xl font-medium ${warn ? 'text-[#b06000]' : 'text-[#202124]'}`}>{value}</p>
      <p className={`text-xs mt-1 ${warn ? 'text-[#b06000]' : 'text-[#5f6368]'}`}>{sub}</p>
    </div>
  );
}

const SIDE_ICONS: Record<string, React.ReactElement> = {
  dashboard: <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z" />,
  analyze:   <path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z" />,
  model:     <path d="M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0l4.6-4.6-4.6-4.6L16 6l6 6-6 6-1.4-1.4z" />,
  billing:   <path d="M20 4H4c-1.11 0-2 .89-2 2v12c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V6c0-1.11-.89-2-2-2zm0 14H4v-6h16v6zm0-10H4V6h16v2z" />,
  docs:      <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" />,
  faq:       <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z" />,
};

function SideLink({ href, icon, label, active }: { href: string; icon: string; label: string; active?: boolean }) {
  return (
    <Link
      href={href}
      className={`flex items-center gap-3 px-4 h-11 rounded-r-full text-sm transition-colors ${
        active
          ? 'bg-[#e8f0fe] text-[#1a73e8] font-medium'
          : 'text-[#3c4043] hover:bg-[#f1f3f4]'
      }`}
    >
      <svg className="w-5 h-5 shrink-0" fill="currentColor" viewBox="0 0 24 24">
        {SIDE_ICONS[icon]}
      </svg>
      <span dangerouslySetInnerHTML={{ __html: label }} />
    </Link>
  );
}
