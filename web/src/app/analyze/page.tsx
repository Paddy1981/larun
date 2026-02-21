'use client';

import { useState, useEffect, Suspense } from 'react';
import Link from 'next/link';
import { useSession } from 'next-auth/react';
import { useSearchParams } from 'next/navigation';
import Header from '@/components/Header';

interface TICInfo {
  ra: number;
  dec: number;
  tmag: number;
  teff: number;
  radius: number;
  mass: number;
}

interface VettingTest {
  flag: string;
  message: string;
}

interface AnalysisResult {
  id: string;
  tic_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: {
    detection: boolean;
    confidence: number;
    period_days: number | null;
    depth_ppm: number | null;
    duration_hours: number | null;
    epoch_btjd?: number | null;
    snr?: number | null;
    vetting?: {
      disposition: string;
      confidence: number;
      odd_even: VettingTest;
      v_shape: VettingTest;
      secondary_eclipse: VettingTest;
    };
    sectors_used?: number[];
    processing_time_seconds?: number;
    tic_info?: TICInfo;
    folded_curve?: { phase: number; flux: number }[];
  };
  error?: string;
}

const confirmedTargets = [
  { id: '470710327', name: 'TOI-1338 b',  description: 'Circumbinary · 14.6d',       confirmed: true },
  { id: '307210830', name: 'TOI-700 d',   description: 'Earth-sized HZ · 37.4d',     confirmed: true },
  { id: '441462736', name: 'TOI-849 b',   description: 'Dense Neptune · 18.4h',      confirmed: true },
  { id: '141527579', name: 'TOI-561 b',   description: 'Super-Earth · 10.8h',        confirmed: true },
  { id: '261136679', name: 'TOI-175 b',   description: 'Sub-Neptune · 3.7d',         confirmed: true },
  { id: '149603524', name: 'WASP-121 b',  description: 'Ultra-hot Jupiter · 1.3d',   confirmed: true },
  { id: '29960110',  name: 'LTT 9779 b',  description: 'Ultra-hot Neptune · 0.8d',   confirmed: true },
  { id: '271893367', name: 'TOI-132 b',   description: 'Dense Neptune · 2.1d',       confirmed: true },
  { id: '158588995', name: 'TOI-1136 b',  description: '6-planet system · 4.2d',     confirmed: true },
  { id: '395171208', name: 'TOI-4153 b',  description: 'Hot Saturn · 5.0d',          confirmed: true },
];

const candidateTargets = [
  { id: '231702397', name: 'TOI-1231.01', description: 'Sub-Neptune · 24.3d' },
  { id: '396740648', name: 'TOI-2136.01', description: 'Sub-Neptune · 7.9d' },
  { id: '267263253', name: 'TOI-1452.01', description: 'Ocean world? · 11.1d' },
  { id: '150428135', name: 'TOI-1695.01', description: 'Sub-Earth · 3.1d' },
  { id: '219195044', name: 'TOI-1759.01', description: 'M-dwarf · 18.9d' },
  { id: '467179528', name: 'TOI-1266.01', description: 'Sub-Neptune · 10.9d' },
  { id: '455737351', name: 'TOI-2119.01', description: 'M-dwarf · 7.2d' },
  { id: '349488688', name: 'TOI-1806.01', description: 'Sub-Neptune · 6.6d' },
  { id: '237913869', name: 'TOI-1694.01', description: 'Hot Neptune · 3.8d' },
  { id: '394050135', name: 'TOI-2084.01', description: 'M-dwarf · 6.1d' },
  { id: '284361752', name: 'TOI-4342.01', description: 'Earth-sized · 9.1d' },
  { id: '372172128', name: 'TOI-3714.01', description: 'Super-Earth · 2.5d' },
];

// ── Transit chart (SVG, inline, no extra deps) ────────────────────────────
function TransitChart({
  foldedCurve,
}: {
  foldedCurve: { phase: number; flux: number }[];
}) {
  const W = 560, H = 200;
  const PAD = { top: 16, right: 16, bottom: 36, left: 52 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const fluxVals = foldedCurve.map(p => p.flux);
  const rawMin = Math.min(...fluxVals);
  const rawMax = Math.max(...fluxVals);
  const pad = Math.max((rawMax - rawMin) * 0.3, 0.0002);
  const yMin = rawMin - pad;
  const yMax = rawMax + pad;

  const xScale = (phase: number) => PAD.left + (phase + 0.5) * plotW;
  const yScale = (flux: number)  => PAD.top + plotH - ((flux - yMin) / (yMax - yMin)) * plotH;

  const yMid  = yScale(1.0);
  const yTick = (v: number) => `${(v * 1e6 - 1e6).toFixed(0)}`;

  const nTicks = 4;
  const yTicks = Array.from({ length: nTicks + 1 }, (_, i) => yMin + (i / nTicks) * (yMax - yMin));

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full h-auto"
      aria-label="Phase-folded transit light curve"
    >
      {/* Plot background */}
      <rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} fill="#f8f9fa" rx={4} />

      {/* Y grid lines */}
      {yTicks.map((v, i) => (
        <line key={i}
          x1={PAD.left} y1={yScale(v)}
          x2={PAD.left + plotW} y2={yScale(v)}
          stroke="#e8eaed" strokeWidth={1}
        />
      ))}

      {/* Baseline (flux = 1.0) */}
      <line x1={PAD.left} y1={yMid} x2={PAD.left + plotW} y2={yMid}
        stroke="#bdc1c6" strokeWidth={1} strokeDasharray="4 3" />

      {/* Transit center marker */}
      <line x1={xScale(0)} y1={PAD.top} x2={xScale(0)} y2={PAD.top + plotH}
        stroke="#1a73e8" strokeWidth={1} strokeDasharray="4 3" opacity={0.6} />

      {/* Data points */}
      {foldedCurve.map((p, i) => (
        <circle key={i}
          cx={xScale(p.phase)} cy={yScale(p.flux)}
          r={2.5} fill="#1a73e8" opacity={0.75}
        />
      ))}

      {/* Axes */}
      <line x1={PAD.left} y1={PAD.top + plotH} x2={PAD.left + plotW} y2={PAD.top + plotH}
        stroke="#5f6368" strokeWidth={1} />
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH}
        stroke="#5f6368" strokeWidth={1} />

      {/* X axis labels */}
      {[-0.5, -0.25, 0, 0.25, 0.5].map(v => (
        <text key={v} x={xScale(v)} y={H - 6}
          textAnchor="middle" fontSize={10} fill="#5f6368">{v}</text>
      ))}
      <text x={W / 2} y={H - 0} textAnchor="middle" fontSize={10} fill="#5f6368">
        Orbital Phase
      </text>

      {/* Y axis labels */}
      {yTicks.slice(0, -1).map((v, i) => (
        <text key={i} x={PAD.left - 4} y={yScale(v) + 4}
          textAnchor="end" fontSize={9} fill="#5f6368">{yTick(v)}</text>
      ))}
      <text
        x={12} y={PAD.top + plotH / 2}
        textAnchor="middle" fontSize={10} fill="#5f6368"
        transform={`rotate(-90,12,${PAD.top + plotH / 2})`}
      >
        Flux (ppm)
      </text>

      {/* Transit label */}
      <text x={xScale(0)} y={PAD.top + 10}
        textAnchor="middle" fontSize={9} fill="#1a73e8">Transit</text>
    </svg>
  );
}

// ── DSS sky image ─────────────────────────────────────────────────────────
function SkyImage({ ra, dec, size = 5 }: { ra: number; dec: number; size?: number }) {
  const url = `https://archive.stsci.edu/cgi-bin/dss_search?v=poss2ukstu_red&r=${ra}&d=${dec}&e=J2000&h=${size}&w=${size}&f=gif&c=none`;
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={url}
      alt={`DSS sky field around RA ${ra.toFixed(3)} Dec ${dec.toFixed(3)}`}
      className="w-full rounded-lg object-cover"
      style={{ aspectRatio: '1/1', background: '#0a0a14' }}
      onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }}
    />
  );
}

// ── Planet radius estimate ─────────────────────────────────────────────────
function estimatePlanetRadius(depthPpm: number, stellarRadiusSun: number): {
  radiusEarth: number; radiusJupiter: number;
} {
  const fracDepth = depthPpm / 1e6;
  const rpRs = Math.sqrt(fracDepth);          // Rp/R* ratio
  const rSunInEarth = 109.076;                 // 1 R_sun = 109.076 R_earth
  const rSunInJupiter = 9.9317;                // 1 R_sun = 9.9317 R_Jupiter
  return {
    radiusEarth:   rpRs * stellarRadiusSun * rSunInEarth,
    radiusJupiter: rpRs * stellarRadiusSun * rSunInJupiter,
  };
}

// ── Vetting flag chip ─────────────────────────────────────────────────────
function FlagChip({ flag }: { flag: string }) {
  const styles: Record<string, string> = {
    PASS:    'bg-green-100 text-green-700',
    WARNING: 'bg-yellow-100 text-yellow-700',
    FAIL:    'bg-red-100 text-red-700',
  };
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${styles[flag] ?? 'bg-gray-100 text-gray-700'}`}>
      {flag}
    </span>
  );
}

interface LiveCandidate {
  tic_id: string;
  toi: string;
  period_days: number;
  depth_ppm: number;
  duration_hours: number;
  disposition: string;
  planet_radius_earth: number | null;
  star_tmag: number | null;
}

// ═══════════════════════════════════════════════════════════════════════════
function AnalyzePageInner() {
  const { status } = useSession();
  const searchParams = useSearchParams();
  const [ticId, setTicId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [liveCandidates, setLiveCandidates] = useState<LiveCandidate[]>([]);
  const [candidatesLoading, setCandidatesLoading] = useState(true);
  const [showAllLive, setShowAllLive] = useState(false);

  // Fetch live TOI candidates from NASA Exoplanet Archive (via our API)
  useEffect(() => {
    fetch('/api/v1/toi-candidates')
      .then(r => r.json())
      .then(d => {
        if (d.candidates) setLiveCandidates(d.candidates);
      })
      .catch(() => {/* silent – hardcoded fallback still shown */})
      .finally(() => setCandidatesLoading(false));
  }, []);

  // Pre-fill TIC ID from ?tic= query param and auto-start
  useEffect(() => {
    const tic = searchParams.get('tic');
    if (tic && status !== 'loading' && !isLoading && !result) {
      setTicId(tic);
      handleAnalyze(tic);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, searchParams]);

  const handleAnalyze = async (targetId?: string) => {
    const idToAnalyze = (targetId ?? ticId).trim();
    if (!idToAnalyze) { setError('Please enter a TIC ID'); return; }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);

    const tick = setInterval(() => setProgress(p => Math.min(85, p + 4)), 1500);

    try {
      const response = await fetch('/api/v1/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tic_id: idToAnalyze }),
      });

      clearInterval(tick);

      const text = await response.text();
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(
          response.status === 504
            ? 'Analysis timed out fetching TESS data. Try again or choose a different target.'
            : `Server error (${response.status}). Please try again.`
        );
      }

      if (!response.ok) {
        throw new Error((data.error as { message?: string })?.message ?? 'Failed to start analysis');
      }

      setProgress(100);
      setResult({
        id:     data.analysis_id as string,
        tic_id: data.tic_id as string,
        status: data.status  as AnalysisResult['status'],
        result: data.result  as AnalysisResult['result'],
        error:  data.error   as string | undefined,
      });
    } catch (err) {
      clearInterval(tick);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => { setResult(null); setTicId(''); setError(null); setProgress(0); };

  // Loading auth
  if (status === 'loading') {
    return (
      <div className="min-h-screen flex flex-col bg-white">
        <Header />
        <main className="flex-1 pt-24 pb-16 flex items-center justify-center">
          <div className="w-10 h-10 border-4 border-[#1a73e8] border-t-transparent rounded-full animate-spin" />
        </main>
      </div>
    );
  }

  const res = result?.result;
  const detected = res?.detection === true;

  // ── Planet size estimate ───────────────────────────────────────────────
  const planetSize = detected && res?.depth_ppm && res?.tic_info?.radius
    ? estimatePlanetRadius(res.depth_ppm, res.tic_info.radius)
    : null;

  return (
    <div className="min-h-screen flex flex-col bg-white">
      <Header />

      <main className="flex-1 pt-24 pb-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* ── Title ─────────────────────────────────────────────────── */}
          <div className="text-center mb-10">
            <h1 className="text-3xl font-bold text-[#202124] mb-3">Analyze Target</h1>
            <p className="text-[#5f6368] max-w-2xl mx-auto">
              Enter a TESS Input Catalog (TIC) ID to search for exoplanet transit signals
              using BLS transit detection on real TESS light curves.
            </p>
          </div>

          {/* ── Search form ────────────────────────────────────────────── */}
          <div className="bg-white border border-[#dadce0] rounded-xl p-6 mb-6 shadow-sm">
            <div className="flex gap-3">
              <input
                type="text"
                placeholder="Enter TIC ID (e.g., 470710327)"
                value={ticId}
                onChange={e => setTicId(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !isLoading && ticId.trim() && handleAnalyze()}
                disabled={isLoading}
                className="flex-1 px-4 py-3 border border-[#dadce0] rounded-lg text-[#202124] placeholder-[#9aa0a6] focus:outline-none focus:ring-2 focus:ring-[#1a73e8] focus:border-transparent disabled:bg-[#f1f3f4] disabled:cursor-not-allowed"
              />
              <button
                onClick={() => handleAnalyze()}
                disabled={isLoading || !ticId.trim()}
                className="px-6 py-3 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors disabled:bg-[#dadce0] disabled:cursor-not-allowed"
              >
                {isLoading ? 'Analyzing…' : 'Analyze'}
              </button>
            </div>
            {error && (
              <p className="mt-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2">
                {error}
              </p>
            )}
          </div>

          {/* ── Target pickers ─────────────────────────────────────────── */}
          {!isLoading && !result && (
            <div className="mb-8 space-y-5">

              {/* Confirmed planets */}
              <div>
                <p className="text-xs font-semibold text-[#5f6368] uppercase tracking-wider mb-2">
                  Confirmed Planets
                </p>
                <div className="flex flex-wrap gap-2">
                  {confirmedTargets.map(t => (
                    <button key={t.id}
                      onClick={() => { setTicId(t.id); handleAnalyze(t.id); }}
                      className="px-3 py-1.5 bg-[#e6f4ea] hover:bg-[#ceead6] text-[#137333] text-xs font-medium rounded-full transition-colors"
                    >
                      {t.name} · {t.description}
                    </button>
                  ))}
                </div>
              </div>

              {/* Static curated candidates */}
              <div>
                <p className="text-xs font-semibold text-[#5f6368] uppercase tracking-wider mb-2">
                  Curated Candidates
                </p>
                <div className="flex flex-wrap gap-2">
                  {candidateTargets.map(t => (
                    <button key={t.id}
                      onClick={() => { setTicId(t.id); handleAnalyze(t.id); }}
                      className="px-3 py-1.5 bg-[#fef7e0] hover:bg-[#fde293] text-[#b06000] text-xs font-medium rounded-full transition-colors"
                    >
                      {t.name} · {t.description}
                    </button>
                  ))}
                </div>
              </div>

              {/* Live TOI candidates from NASA Exoplanet Archive */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <p className="text-xs font-semibold text-[#5f6368] uppercase tracking-wider">
                    Live TOI Catalog
                  </p>
                  <span className="px-2 py-0.5 bg-[#e8f0fe] text-[#1a73e8] text-xs rounded-full font-medium">
                    NASA Exoplanet Archive
                  </span>
                  {candidatesLoading && (
                    <span className="w-3 h-3 border-2 border-[#1a73e8] border-t-transparent rounded-full animate-spin inline-block" />
                  )}
                  {!candidatesLoading && liveCandidates.length > 0 && (
                    <span className="text-xs text-[#5f6368]">{liveCandidates.length} targets</span>
                  )}
                </div>
                {!candidatesLoading && liveCandidates.length === 0 && (
                  <p className="text-xs text-[#9aa0a6]">Could not load live catalog — using curated list above.</p>
                )}
                <div className="flex flex-wrap gap-2">
                  {(showAllLive ? liveCandidates : liveCandidates.slice(0, 30)).map(c => (
                    <button key={c.tic_id}
                      onClick={() => { setTicId(c.tic_id); handleAnalyze(c.tic_id); }}
                      title={`TIC ${c.tic_id} · ${c.period_days.toFixed(2)}d · depth ${(c.depth_ppm / 1e4).toFixed(2)}% · ${c.disposition}`}
                      className="px-3 py-1.5 bg-[#f0f4ff] hover:bg-[#d2e3fc] text-[#174ea6] text-xs font-medium rounded-full transition-colors"
                    >
                      TOI-{c.toi} · {c.period_days.toFixed(1)}d
                    </button>
                  ))}
                </div>
                {!candidatesLoading && liveCandidates.length > 30 && (
                  <button
                    onClick={() => setShowAllLive(v => !v)}
                    className="mt-2 text-xs text-[#1a73e8] hover:underline"
                  >
                    {showAllLive ? 'Show fewer' : `Show all ${liveCandidates.length} targets`}
                  </button>
                )}
              </div>

            </div>
          )}

          {/* ── Loading ────────────────────────────────────────────────── */}
          {isLoading && (
            <div className="bg-white border border-[#dadce0] rounded-xl p-8 text-center shadow-sm">
              <div className="w-16 h-16 mx-auto mb-4 relative">
                <div className="absolute inset-0 border-4 border-[#f1f3f4] rounded-full" />
                <div className="absolute inset-0 border-4 border-[#1a73e8] rounded-full border-t-transparent animate-spin" />
              </div>
              <h3 className="text-lg font-medium text-[#202124] mb-1">Analyzing TIC {ticId || 'target'}…</h3>
              <p className="text-sm text-[#5f6368] mb-4">Fetching TESS data · Running BLS detection · Vetting</p>
              <div className="w-full bg-[#f1f3f4] rounded-full h-2 mb-2">
                <div className="bg-[#1a73e8] h-2 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }} />
              </div>
              <p className="text-xs text-[#5f6368]">{progress}%</p>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════
               RESULTS — Detection found
          ══════════════════════════════════════════════════════════════ */}
          {result && result.status === 'completed' && res && (
            <div className="space-y-6">

              {/* ── Status banner ──────────────────────────────────────── */}
              <div className={`rounded-xl px-6 py-4 flex items-center gap-4 ${
                detected
                  ? 'bg-green-50 border border-green-200'
                  : 'bg-[#f1f3f4] border border-[#dadce0]'
              }`}>
                <div className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 ${
                  detected ? 'bg-green-100' : 'bg-[#dadce0]'
                }`}>
                  {detected ? (
                    <svg className="w-7 h-7 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <svg className="w-7 h-7 text-[#5f6368]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  )}
                </div>
                <div>
                  <h2 className={`text-xl font-bold ${detected ? 'text-green-700' : 'text-[#202124]'}`}>
                    {detected ? 'Transit Signal Detected!' : 'No Transit Signal Detected'}
                  </h2>
                  <p className="text-[#5f6368] text-sm">
                    TIC {result.tic_id}
                    {res.sectors_used && res.sectors_used.length > 0 &&
                      ` · Sector${res.sectors_used.length > 1 ? 's' : ''} ${res.sectors_used.join(', ')}`}
                    {res.processing_time_seconds &&
                      ` · ${res.processing_time_seconds.toFixed(1)}s`}
                  </p>
                </div>
              </div>

              {detected && (
                <>
                  {/* ── Top row: sky image + metrics ──────────────────── */}
                  <div className="grid md:grid-cols-5 gap-6">

                    {/* Sky image (DSS) */}
                    {res.tic_info?.ra !== undefined && (
                      <div className="md:col-span-2 bg-[#0a0a14] rounded-xl overflow-hidden">
                        <SkyImage ra={res.tic_info.ra} dec={res.tic_info.dec} size={5} />
                        <div className="px-3 py-2 text-center">
                          <p className="text-xs text-[#9aa0a6]">
                            DSS Red · 5′ field · RA {res.tic_info.ra.toFixed(3)} Dec {res.tic_info.dec.toFixed(3)}
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Metrics grid */}
                    <div className={`${res.tic_info?.ra !== undefined ? 'md:col-span-3' : 'md:col-span-5'} grid grid-cols-2 gap-3 content-start`}>
                      {[
                        { label: 'Confidence',   value: `${(res.confidence * 100).toFixed(1)}%`,                          sub: res.vetting?.disposition?.replace(/_/g, ' ') ?? '' },
                        { label: 'Period',        value: res.period_days?.toFixed(3) ?? '—',                              sub: 'days' },
                        { label: 'Transit Depth', value: res.depth_ppm?.toFixed(0) ?? '—',                               sub: 'ppm' },
                        { label: 'Duration',      value: res.duration_hours?.toFixed(2) ?? '—',                          sub: 'hours' },
                        { label: 'SNR',           value: res.snr?.toFixed(1) ?? '—',                                    sub: 'signal-to-noise' },
                        { label: 'Epoch',         value: res.epoch_btjd ? `BTJD ${res.epoch_btjd.toFixed(2)}` : '—',   sub: 'first transit' },
                      ].map(({ label, value, sub }) => (
                        <div key={label} className="bg-white border border-[#dadce0] rounded-xl p-4">
                          <p className="text-xs font-semibold text-[#5f6368] uppercase tracking-wider mb-1">{label}</p>
                          <p className="text-2xl font-bold text-[#202124] leading-tight">{value}</p>
                          {sub && <p className="text-xs text-[#9aa0a6] mt-0.5">{sub}</p>}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* ── Planet size estimate ─────────────────────────── */}
                  {planetSize && (
                    <div className="bg-[#e8f0fe] border border-[#c5d8fb] rounded-xl p-5 flex items-center gap-6">
                      <div className="flex-shrink-0">
                        {/* Planet icon: concentric circles */}
                        <svg width={60} height={60} viewBox="0 0 60 60">
                          <circle cx={30} cy={30} r={28} fill="#1a73e8" opacity={0.12} />
                          <circle cx={30} cy={30} r={18} fill="#1a73e8" opacity={0.25} />
                          <circle cx={30} cy={30} r={10} fill="#1a73e8" opacity={0.6} />
                          <ellipse cx={30} cy={30} rx={28} ry={8} fill="none" stroke="#1a73e8" strokeWidth={1.5} opacity={0.4} />
                        </svg>
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-[#1557b0] mb-1">Estimated Planet Size</p>
                        <p className="text-2xl font-bold text-[#1a73e8]">
                          {planetSize.radiusEarth.toFixed(2)} R<sub>⊕</sub>
                        </p>
                        <p className="text-sm text-[#5f6368]">
                          {planetSize.radiusJupiter.toFixed(3)} R<sub>J</sub>
                          {' · '}
                          {planetSize.radiusEarth < 1.6 ? 'Rocky / Super-Earth' :
                           planetSize.radiusEarth < 4   ? 'Sub-Neptune' :
                           planetSize.radiusEarth < 10  ? 'Neptune-class' :
                           'Jupiter-class'}
                          {res.period_days && ` · ${res.period_days.toFixed(1)}-day orbit`}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* ── Phase-folded light curve ──────────────────────── */}
                  {res.folded_curve && res.folded_curve.length > 0 && (
                    <div className="bg-white border border-[#dadce0] rounded-xl p-6 shadow-sm">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h3 className="text-lg font-semibold text-[#202124]">Phase-Folded Light Curve</h3>
                          <p className="text-xs text-[#5f6368]">
                            TESS photometry folded at {res.period_days?.toFixed(4)} d — transit centred at phase 0
                          </p>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-[#5f6368]">
                          <span className="inline-block w-3 h-3 rounded-full bg-[#1a73e8] opacity-75" />
                          Binned flux
                        </div>
                      </div>
                      <TransitChart foldedCurve={res.folded_curve} />
                    </div>
                  )}

                  {/* ── Stellar properties ───────────────────────────── */}
                  {res.tic_info && (
                    <div className="bg-white border border-[#dadce0] rounded-xl p-6 shadow-sm">
                      <h3 className="text-lg font-semibold text-[#202124] mb-4">Host Star Properties</h3>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        {[
                          { label: 'Teff',    value: res.tic_info.teff ? `${res.tic_info.teff.toFixed(0)} K`  : '—', sub: 'Effective temperature' },
                          { label: 'Radius',  value: res.tic_info.radius ? `${res.tic_info.radius.toFixed(2)} R☉` : '—', sub: 'Stellar radius' },
                          { label: 'Mass',    value: res.tic_info.mass ? `${res.tic_info.mass.toFixed(2)} M☉` : '—',   sub: 'Stellar mass' },
                          { label: 'T mag',   value: res.tic_info.tmag ? res.tic_info.tmag.toFixed(2) : '—',           sub: 'TESS magnitude' },
                        ].map(({ label, value, sub }) => (
                          <div key={label} className="bg-[#f8f9fa] rounded-lg p-3">
                            <p className="text-xs font-semibold text-[#5f6368] uppercase tracking-wider mb-1">{label}</p>
                            <p className="text-lg font-bold text-[#202124]">{value}</p>
                            <p className="text-xs text-[#9aa0a6]">{sub}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* ── Vetting tests ─────────────────────────────────── */}
                  {res.vetting && (
                    <div className="bg-white border border-[#dadce0] rounded-xl p-6 shadow-sm">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-[#202124]">Vetting Tests</h3>
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                          res.vetting.disposition === 'PLANET_CANDIDATE'    ? 'bg-green-100 text-green-700'
                          : res.vetting.disposition === 'LIKELY_FALSE_POSITIVE' ? 'bg-red-100 text-red-700'
                          : 'bg-yellow-100 text-yellow-700'
                        }`}>
                          {res.vetting.disposition.replace(/_/g, ' ')}
                        </span>
                      </div>
                      <div className="space-y-3">
                        {[
                          { name: 'Odd–Even Depth Test',    test: res.vetting.odd_even },
                          { name: 'V-Shape Analysis',       test: res.vetting.v_shape },
                          { name: 'Secondary Eclipse Check',test: res.vetting.secondary_eclipse },
                        ].map(({ name, test }) => (
                          <div key={name} className="flex items-center justify-between p-4 bg-[#f8f9fa] rounded-lg">
                            <div className="flex-1 min-w-0 pr-4">
                              <p className="font-medium text-[#202124] text-sm">{name}</p>
                              <p className="text-xs text-[#5f6368] mt-0.5 truncate">{test.message}</p>
                            </div>
                            <FlagChip flag={test.flag} />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* ── No detection: soft message ────────────────────────── */}
              {!detected && (
                <div className="bg-white border border-[#dadce0] rounded-xl p-6 shadow-sm text-[#5f6368] text-sm">
                  <p className="mb-2">
                    No periodic transit signal above the detection threshold (SNR ≥ 7) was found
                    in the available TESS data for TIC {result.tic_id}.
                  </p>
                  <p>
                    This could mean the target has no transiting planet, the planet is not in
                    a favourable orbital geometry, or more sectors are needed. Try a confirmed
                    target to validate the model.
                  </p>
                </div>
              )}

              {/* ── Actions ─────────────────────────────────────────── */}
              <div className="flex gap-4">
                <button onClick={reset}
                  className="px-5 py-2.5 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium rounded-lg transition-colors text-sm">
                  Analyze Another
                </button>
                <Link href="/dashboard"
                  className="px-5 py-2.5 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors text-sm">
                  View Dashboard
                </Link>
              </div>
            </div>
          )}

          {/* ── Failed state ────────────────────────────────────────── */}
          {result && result.status === 'failed' && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-red-700 mb-2">Analysis Failed</h3>
              <p className="text-[#5f6368] text-sm mb-4">
                {result.error ?? `Unable to analyze TIC ${result.tic_id}.`}
              </p>
              <button onClick={reset}
                className="px-5 py-2.5 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors text-sm">
                Try Again
              </button>
            </div>
          )}

          {/* ── How it works ────────────────────────────────────────── */}
          {!isLoading && !result && (
            <div className="mt-12 bg-[#f8f9fa] rounded-xl p-6">
              <h3 className="text-lg font-semibold text-[#202124] mb-4">How It Works</h3>
              <div className="grid md:grid-cols-3 gap-6">
                {[
                  { n: 1, title: 'Data Retrieval',    desc: 'TESS light curves fetched from NASA MAST archive using real FITS photometry data' },
                  { n: 2, title: 'BLS Detection',      desc: 'Box Least Squares algorithm searches all periods 0.5–40 days for periodic transit dips' },
                  { n: 3, title: 'Vetting & Scoring',  desc: 'Odd-even, V-shape, and secondary eclipse tests reject false positives' },
                ].map(({ n, title, desc }) => (
                  <div key={n} className="flex gap-3">
                    <div className="w-8 h-8 bg-[#1a73e8] text-white rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0">
                      {n}
                    </div>
                    <div>
                      <p className="font-medium text-[#202124] mb-0.5">{title}</p>
                      <p className="text-sm text-[#5f6368]">{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="py-8 bg-[#f1f3f4] border-t border-[#dadce0]">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-sm text-[#5f6368]">
            &copy; {new Date().getFullYear()} Larun Engineering · Data: NASA MAST / TESS
          </p>
        </div>
      </footer>
    </div>
  );
}

export default function AnalyzePage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="w-8 h-8 border-2 border-[#1a73e8] border-t-transparent rounded-full animate-spin" />
      </div>
    }>
      <AnalyzePageInner />
    </Suspense>
  );
}
