'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import Header from '@/components/Header';

interface VettingTest {
  flag: 'PASS' | 'FAIL' | 'WARNING';
  message: string;
  confidence?: number;
}

interface TICInfo {
  ra: number;
  dec: number;
  tmag: number;
  teff: number;
  radius: number;
  mass: number;
}

interface AnalysisResult {
  id: string;
  tic_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  result?: {
    detection: boolean;
    confidence: number;
    period_days: number | null;
    depth_ppm: number | null;
    duration_hours: number | null;
    epoch_btjd?: number | null;
    snr?: number | null;
    sectors_used?: number[];
    processing_time_seconds?: number;
    folded_curve?: { phase: number; flux: number }[];
    tic_info?: TICInfo;
    vetting?: {
      disposition: string;
      confidence: number;
      odd_even: VettingTest;
      v_shape: VettingTest;
      secondary_eclipse: VettingTest;
    };
  };
  error?: string;
}

// ── Transit chart ──────────────────────────────────────────────────────────
function TransitChart({ foldedCurve }: { foldedCurve: { phase: number; flux: number }[] }) {
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
  const yScale = (flux: number) => PAD.top + plotH - ((flux - yMin) / (yMax - yMin)) * plotH;
  const yMid = yScale(1.0);
  const nTicks = 4;
  const yTicks = Array.from({ length: nTicks + 1 }, (_, i) => yMin + (i / nTicks) * (yMax - yMin));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" aria-label="Phase-folded transit light curve">
      <rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} fill="#f8f9fa" rx={4} />
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD.left} y1={yScale(v)} x2={PAD.left + plotW} y2={yScale(v)} stroke="#e8eaed" strokeWidth={1} />
      ))}
      <line x1={PAD.left} y1={yMid} x2={PAD.left + plotW} y2={yMid} stroke="#bdc1c6" strokeWidth={1} strokeDasharray="4 3" />
      <line x1={xScale(0)} y1={PAD.top} x2={xScale(0)} y2={PAD.top + plotH} stroke="#1a73e8" strokeWidth={1} strokeDasharray="4 3" opacity={0.6} />
      {foldedCurve.map((p, i) => (
        <circle key={i} cx={xScale(p.phase)} cy={yScale(p.flux)} r={2.5} fill="#1a73e8" opacity={0.75} />
      ))}
      <line x1={PAD.left} y1={PAD.top + plotH} x2={PAD.left + plotW} y2={PAD.top + plotH} stroke="#5f6368" strokeWidth={1} />
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH} stroke="#5f6368" strokeWidth={1} />
      {[-0.5, -0.25, 0, 0.25, 0.5].map(v => (
        <text key={v} x={xScale(v)} y={H - 6} textAnchor="middle" fontSize={10} fill="#5f6368">{v}</text>
      ))}
      <text x={W / 2} y={H} textAnchor="middle" fontSize={10} fill="#5f6368">Orbital Phase</text>
      {yTicks.slice(0, -1).map((v, i) => (
        <text key={i} x={PAD.left - 4} y={yScale(v) + 4} textAnchor="end" fontSize={9} fill="#5f6368">
          {`${((v - 1) * 1e6).toFixed(0)}`}
        </text>
      ))}
      <text x={12} y={PAD.top + plotH / 2} textAnchor="middle" fontSize={10} fill="#5f6368"
        transform={`rotate(-90,12,${PAD.top + plotH / 2})`}>Flux (ppm)</text>
      <text x={xScale(0)} y={PAD.top + 10} textAnchor="middle" fontSize={9} fill="#1a73e8">Transit</text>
    </svg>
  );
}

// ── DSS sky image ──────────────────────────────────────────────────────────
function SkyImage({ ra, dec }: { ra: number; dec: number }) {
  const url = `https://archive.stsci.edu/cgi-bin/dss_search?v=poss2ukstu_red&r=${ra}&d=${dec}&e=J2000&h=5&w=5&f=gif&c=none`;
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img src={url} alt={`DSS sky field`} className="w-full rounded-lg object-cover"
      style={{ aspectRatio: '1/1', background: '#0a0a14' }}
      onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }} />
  );
}

// ── Vetting flag chip ──────────────────────────────────────────────────────
function FlagChip({ flag }: { flag: string }) {
  const styles: Record<string, string> = {
    PASS: 'bg-green-100 text-green-700',
    WARNING: 'bg-yellow-100 text-yellow-700',
    FAIL: 'bg-red-100 text-red-700',
  };
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${styles[flag] ?? 'bg-gray-100 text-gray-700'}`}>
      {flag}
    </span>
  );
}

// ── Planet radius estimate ─────────────────────────────────────────────────
function estimatePlanetRadius(depthPpm: number, stellarRadiusSun: number) {
  const rpRs = Math.sqrt(depthPpm / 1e6);
  return {
    radiusEarth: rpRs * stellarRadiusSun * 109.076,
    radiusJupiter: rpRs * stellarRadiusSun * 9.9317,
  };
}

// ══════════════════════════════════════════════════════════════════════════
export default function ResultsPage() {
  const params = useParams();
  const analysisId = params.id as string;

  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const downloadJSON = () => {
    if (!analysis?.result) return;
    const payload = {
      analysis_id: analysis.id,
      tic_id: analysis.tic_id,
      created_at: analysis.created_at,
      status: analysis.status,
      result: analysis.result,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `TIC-${analysis.tic_id}-analysis.json`; a.click();
    URL.revokeObjectURL(url);
  };

  const downloadCSV = () => {
    if (!analysis?.result) return;
    const r = analysis.result;
    const rows = [
      ['Field', 'Value'],
      ['TIC ID', analysis.tic_id],
      ['Analysis ID', analysis.id],
      ['Date', analysis.created_at],
      ['Detection', String(r.detection)],
      ['Confidence', r.confidence != null ? (r.confidence * 100).toFixed(2) + '%' : ''],
      ['Period (days)', r.period_days ?? ''],
      ['Depth (ppm)', r.depth_ppm ?? ''],
      ['Duration (hours)', r.duration_hours ?? ''],
      ['SNR', r.snr ?? ''],
      ['Epoch BTJD', r.epoch_btjd ?? ''],
      ['Sectors used', r.sectors_used?.join(';') ?? ''],
      ['Vetting disposition', r.vetting?.disposition ?? ''],
      ['Vetting confidence', r.vetting?.confidence != null ? (r.vetting.confidence * 100).toFixed(2) + '%' : ''],
      ['Odd-even', r.vetting?.odd_even?.flag ?? ''],
      ['V-shape', r.vetting?.v_shape?.flag ?? ''],
      ['Secondary eclipse', r.vetting?.secondary_eclipse?.flag ?? ''],
      ['Star Teff (K)', r.tic_info?.teff ?? ''],
      ['Star radius (R☉)', r.tic_info?.radius ?? ''],
      ['Star mass (M☉)', r.tic_info?.mass ?? ''],
      ['Star Tmag', r.tic_info?.tmag ?? ''],
    ];
    const csv = rows.map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `TIC-${analysis.tic_id}-analysis.csv`; a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    let pollTimer: ReturnType<typeof setTimeout> | null = null;

    const fetchAnalysis = async () => {
      try {
        const res = await fetch(`/api/v1/analyze/${analysisId}`);
        if (!res.ok) throw new Error('Analysis not found');
        const data = await res.json();
        setAnalysis(data);
        if (data.status === 'pending' || data.status === 'processing') {
          pollTimer = setTimeout(fetchAnalysis, 3000);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load analysis');
      } finally {
        setIsLoading(false);
      }
    };

    if (analysisId) fetchAnalysis();
    return () => { if (pollTimer) clearTimeout(pollTimer); };
  }, [analysisId]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col bg-white">
        <Header />
        <main className="flex-1 pt-24 flex items-center justify-center">
          <div className="w-10 h-10 border-4 border-[#1a73e8] border-t-transparent rounded-full animate-spin" />
        </main>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="min-h-screen flex flex-col bg-white">
        <Header />
        <main className="flex-1 pt-24 pb-16 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="w-16 h-16 bg-red-50 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </div>
            <h2 className="text-xl font-bold text-[#202124] mb-2">Analysis Not Found</h2>
            <p className="text-[#5f6368] mb-6">{error || 'The requested analysis could not be found.'}</p>
            <Link href="/dashboard"
              className="px-5 py-2.5 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors text-sm">
              Back to Dashboard
            </Link>
          </div>
        </main>
      </div>
    );
  }

  const res = analysis.result;
  const detected = res?.detection === true;
  const planetSize = detected && res?.depth_ppm && res?.tic_info?.radius
    ? estimatePlanetRadius(res.depth_ppm, res.tic_info.radius)
    : null;

  return (
    <div className="min-h-screen flex flex-col bg-white">
      <Header />

      <main className="flex-1 pt-24 pb-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* Breadcrumb */}
          <div className="mb-6 flex items-center gap-2 text-sm text-[#5f6368]">
            <Link href="/dashboard" className="hover:text-[#1a73e8] transition-colors">Dashboard</Link>
            <span>/</span>
            <Link href="/cloud/analyze" className="hover:text-[#1a73e8] transition-colors">Analyze</Link>
            <span>/</span>
            <span className="text-[#202124] font-medium">TIC {analysis.tic_id}</span>
          </div>

          {/* Processing state */}
          {analysis.status === 'processing' && (
            <div className="bg-white border border-[#dadce0] rounded-xl p-8 text-center mb-6 shadow-sm">
              <div className="w-10 h-10 border-4 border-[#1a73e8] border-t-transparent rounded-full animate-spin mx-auto mb-4" />
              <p className="text-[#202124] font-medium">Analysis in progress…</p>
              <p className="text-[#5f6368] text-sm mt-1">Fetching TESS light curve and running BLS detection</p>
            </div>
          )}

          {/* Failed state */}
          {analysis.status === 'failed' && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-6 mb-6">
              <h3 className="text-lg font-semibold text-red-700 mb-2">Analysis Failed</h3>
              <p className="text-[#5f6368] text-sm">{analysis.error ?? `Unable to analyze TIC ${analysis.tic_id}.`}</p>
              <div className="mt-4">
                <Link href="/cloud/analyze"
                  className="px-4 py-2 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium rounded-lg transition-colors">
                  Try Again
                </Link>
              </div>
            </div>
          )}

          {/* Completed result */}
          {analysis.status === 'completed' && res && (
            <div className="space-y-5">

              {/* Detection summary banner */}
              {(() => {
                const disposition = res.vetting?.disposition;
                const isPlanetCandidate = disposition === 'PLANET_CANDIDATE';
                const isFalsePositive = disposition === 'LIKELY_FALSE_POSITIVE';
                const isInconclusive = disposition === 'INCONCLUSIVE';

                // Verdict styling
                const verdictBg = !detected
                  ? 'border-[#dadce0] bg-white'
                  : isPlanetCandidate
                  ? 'border-green-300 bg-green-50'
                  : isFalsePositive
                  ? 'border-red-200 bg-red-50'
                  : isInconclusive
                  ? 'border-yellow-300 bg-yellow-50'
                  : 'border-[#dadce0] bg-white';

                const iconBg = !detected
                  ? 'bg-[#f1f3f4]'
                  : isPlanetCandidate
                  ? 'bg-green-100'
                  : isFalsePositive
                  ? 'bg-red-100'
                  : 'bg-yellow-100';

                const titleColor = !detected
                  ? 'text-[#202124]'
                  : isPlanetCandidate
                  ? 'text-green-700'
                  : isFalsePositive
                  ? 'text-red-700'
                  : 'text-yellow-700';

                const verdictLabel = !detected
                  ? null
                  : isPlanetCandidate
                  ? 'Planet Candidate'
                  : isFalsePositive
                  ? 'Likely False Positive'
                  : isInconclusive
                  ? 'Inconclusive'
                  : null;

                const verdictExplain = !detected
                  ? null
                  : isPlanetCandidate
                  ? 'All vetting checks passed. This signal has the expected shape and behaviour of a genuine transiting planet.'
                  : isFalsePositive
                  ? 'Vetting checks indicate this signal is likely caused by an eclipsing binary star or instrumental artefact, not a planet. The alternating transit depths and V-shaped profile are hallmarks of a stellar false positive.'
                  : isInconclusive
                  ? 'Some vetting checks raised warnings. More data or follow-up observations are needed to confirm or rule out a planet.'
                  : null;

                return (
                  <div className={`border rounded-xl p-6 shadow-sm ${verdictBg}`}>
                    <div className="flex items-start gap-4">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${iconBg}`}>
                        {!detected ? (
                          <svg className="w-6 h-6 text-[#5f6368]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                          </svg>
                        ) : isPlanetCandidate ? (
                          <svg className="w-6 h-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                          </svg>
                        ) : isFalsePositive ? (
                          <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                          </svg>
                        ) : (
                          <svg className="w-6 h-6 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-3 flex-wrap">
                          <h1 className={`text-xl font-bold ${titleColor}`}>
                            {detected ? 'Transit Signal Detected' : 'No Transit Signal Detected'}
                          </h1>
                          {verdictLabel && (
                            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                              isPlanetCandidate ? 'bg-green-100 text-green-700'
                              : isFalsePositive ? 'bg-red-100 text-red-700'
                              : 'bg-yellow-100 text-yellow-700'
                            }`}>
                              {verdictLabel}
                            </span>
                          )}
                        </div>
                        {verdictExplain && (
                          <p className={`text-sm mt-2 font-medium ${
                            isPlanetCandidate ? 'text-green-700'
                            : isFalsePositive ? 'text-red-700'
                            : 'text-yellow-700'
                          }`}>
                            {verdictExplain}
                          </p>
                        )}
                        <p className="text-[#5f6368] text-xs mt-2">
                          TIC {analysis.tic_id}
                          {res.sectors_used && res.sectors_used.length > 0 &&
                            ` · Sector${res.sectors_used.length > 1 ? 's' : ''} ${res.sectors_used.join(', ')}`}
                          {res.processing_time_seconds && ` · ${res.processing_time_seconds.toFixed(1)}s`}
                          {analysis.completed_at && ` · ${new Date(analysis.completed_at).toLocaleDateString()}`}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })()}

              {detected && (
                <>
                  {/* Sky image + metrics */}
                  <div className="grid md:grid-cols-5 gap-5">
                    {res.tic_info?.ra !== undefined && (
                      <div className="md:col-span-2 bg-[#0a0a14] rounded-xl overflow-hidden">
                        <SkyImage ra={res.tic_info.ra} dec={res.tic_info.dec} />
                        <div className="px-3 py-2 text-center">
                          <p className="text-xs text-[#9aa0a6]">
                            DSS Red · RA {res.tic_info.ra.toFixed(3)} Dec {res.tic_info.dec.toFixed(3)}
                          </p>
                        </div>
                      </div>
                    )}
                    <div className={`${res.tic_info?.ra !== undefined ? 'md:col-span-3' : 'md:col-span-5'} grid grid-cols-2 gap-3 content-start`}>
                      {[
                        { label: 'Confidence',    value: `${(res.confidence * 100).toFixed(1)}%`,                        sub: res.vetting?.disposition?.replace(/_/g, ' ') ?? '' },
                        { label: 'Period',         value: res.period_days?.toFixed(3) ?? '—',                            sub: 'days' },
                        { label: 'Transit Depth',  value: res.depth_ppm?.toFixed(0) ?? '—',                             sub: 'ppm' },
                        { label: 'Duration',       value: res.duration_hours?.toFixed(2) ?? '—',                        sub: 'hours' },
                        { label: 'SNR',            value: res.snr?.toFixed(1) ?? '—',                                   sub: 'signal-to-noise' },
                        { label: 'Epoch',          value: res.epoch_btjd ? `BTJD ${res.epoch_btjd.toFixed(2)}` : '—', sub: 'first transit' },
                      ].map(({ label, value, sub }) => (
                        <div key={label} className="bg-white border border-[#dadce0] rounded-xl p-4">
                          <p className="text-xs font-semibold text-[#5f6368] uppercase tracking-wider mb-1">{label}</p>
                          <p className="text-2xl font-bold text-[#202124] leading-tight">{value}</p>
                          {sub && <p className="text-xs text-[#9aa0a6] mt-0.5">{sub}</p>}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Planet size estimate */}
                  {planetSize && (
                    <div className="bg-[#e8f0fe] border border-[#c5d8fb] rounded-xl p-5 flex items-center gap-6">
                      <svg width={60} height={60} viewBox="0 0 60 60" className="flex-shrink-0">
                        <circle cx={30} cy={30} r={28} fill="#1a73e8" opacity={0.12} />
                        <circle cx={30} cy={30} r={18} fill="#1a73e8" opacity={0.25} />
                        <circle cx={30} cy={30} r={10} fill="#1a73e8" opacity={0.6} />
                        <ellipse cx={30} cy={30} rx={28} ry={8} fill="none" stroke="#1a73e8" strokeWidth={1.5} opacity={0.4} />
                      </svg>
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

                  {/* Why this result — rationale */}
                  {(() => {
                    const snr = res.snr ?? 0;
                    const conf = res.confidence ?? 0;
                    const depth = res.depth_ppm ?? 0;
                    const period = res.period_days ?? 0;
                    const vetting = res.vetting;
                    const failCount = vetting
                      ? [vetting.odd_even, vetting.v_shape, vetting.secondary_eclipse].filter(t => t.flag === 'FAIL').length
                      : 0;

                    const confLabel = conf >= 0.85 ? 'very high' : conf >= 0.65 ? 'high' : conf >= 0.45 ? 'moderate' : 'low';
                    const snrLabel = snr >= 15 ? 'strong' : snr >= 10 ? 'clear' : snr >= 7 ? 'marginal' : 'weak';
                    const depthLabel = depth < 500 ? 'shallow' : depth < 5000 ? 'moderate' : 'deep';

                    return (
                      <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5 shadow-sm">
                        <h3 className="text-sm font-semibold text-[#202124] mb-3 flex items-center gap-2">
                          <svg className="w-4 h-4 text-[#1a73e8]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          Why this result?
                        </h3>
                        <ul className="space-y-2 text-sm text-[#5f6368]">
                          <li>
                            <span className="font-medium text-[#202124]">Detection confidence {(conf * 100).toFixed(0)}% ({confLabel}).</span>
                            {' '}The BLS algorithm scored the periodic signal at {(conf * 100).toFixed(1)}% confidence —
                            {conf < 0.5
                              ? ' below the threshold typically required to claim a robust detection.'
                              : conf < 0.7
                              ? ' a moderate score; real planets can fall here but so can stellar variability.'
                              : ' a strong score suggesting the periodicity is real.'}
                          </li>
                          <li>
                            <span className="font-medium text-[#202124]">SNR {snr.toFixed(1)} ({snrLabel} signal).</span>
                            {' '}Signal-to-noise ratio measures how far the transit dip stands above the noise floor.
                            {snr < 7
                              ? ' Below 7 the signal is marginal and likely noise.'
                              : snr < 10
                              ? ' Between 7–10 is detectable but at the edge of reliability.'
                              : ' Above 10 is a clear, statistically significant dip.'}
                          </li>
                          <li>
                            <span className="font-medium text-[#202124]">Transit depth {depth.toFixed(0)} ppm ({depthLabel}).</span>
                            {' '}Depth is the fraction of starlight blocked per transit (1 ppm = 0.0001%).
                            {depth > 10000
                              ? ' Depths above 1% are unusual for planets — large eclipsing binaries produce this.'
                              : depth > 2000
                              ? ' This depth implies a Jupiter-class or larger body, which can also be a stellar companion.'
                              : ' Consistent with a Neptune- to Jupiter-sized planet.'}
                          </li>
                          {period > 0 && (
                            <li>
                              <span className="font-medium text-[#202124]">Orbital period {period.toFixed(3)} days.</span>
                              {' '}BLS found the most significant periodic signal at this period.
                              {period < 1
                              ? ' Ultra-short periods (< 1 d) can be artefacts; confirm with additional sectors.'
                              : period < 10
                              ? ' Short-period planets are commonly confirmed in TESS data.'
                              : ' Longer periods require more transits — results are less certain with few TESS sectors.'}
                            </li>
                          )}
                          {vetting && (
                            <li>
                              <span className="font-medium text-[#202124]">
                                {failCount} of 3 vetting tests failed.
                              </span>
                              {' '}
                              {failCount === 0
                                ? 'No false-positive indicators found — consistent with a genuine planet.'
                                : failCount === 1
                                ? 'One warning raised; may still be a planet but warrants caution.'
                                : `${failCount} tests flagged characteristics typical of an eclipsing binary or stellar artefact, driving the "Likely False Positive" verdict.`}
                            </li>
                          )}
                        </ul>
                      </div>
                    );
                  })()}

                  {/* Phase-folded light curve */}
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

                  {/* Stellar properties */}
                  {res.tic_info && (
                    <div className="bg-white border border-[#dadce0] rounded-xl p-6 shadow-sm">
                      <h3 className="text-lg font-semibold text-[#202124] mb-4">Host Star Properties</h3>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        {[
                          { label: 'Teff',   value: res.tic_info.teff   ? `${res.tic_info.teff.toFixed(0)} K`   : '—', sub: 'Effective temperature' },
                          { label: 'Radius', value: res.tic_info.radius ? `${res.tic_info.radius.toFixed(2)} R☉` : '—', sub: 'Stellar radius' },
                          { label: 'Mass',   value: res.tic_info.mass   ? `${res.tic_info.mass.toFixed(2)} M☉`   : '—', sub: 'Stellar mass' },
                          { label: 'T mag',  value: res.tic_info.tmag   ? res.tic_info.tmag.toFixed(2)            : '—', sub: 'TESS magnitude' },
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

                  {/* Vetting tests */}
                  {res.vetting && (
                    <div className="bg-white border border-[#dadce0] rounded-xl p-6 shadow-sm">
                      <div className="mb-4">
                        <h3 className="text-lg font-semibold text-[#202124]">False Positive Vetting Tests</h3>
                        <p className="text-xs text-[#5f6368] mt-1">
                          These checks look for signs that the transit signal is caused by a background eclipsing
                          binary or other stellar artefact rather than a genuine planet.
                          A <span className="font-semibold text-red-600">FAIL</span> means the signal shows that suspicious
                          characteristic; a <span className="font-semibold text-green-600">PASS</span> means it does not.
                        </p>
                      </div>
                      <div className="space-y-3">
                        {[
                          {
                            name: 'Odd–Even Depth Test',
                            hint: 'In a real planet transit every dip should be identical. Different depths on alternating transits suggest two stars orbiting each other.',
                            test: res.vetting.odd_even,
                          },
                          {
                            name: 'V-Shape Analysis',
                            hint: 'Real planet transits have a flat bottom. A pure V-shape indicates a grazing or stellar eclipse rather than a disc-like planet blocking the star.',
                            test: res.vetting.v_shape,
                          },
                          {
                            name: 'Secondary Eclipse Check',
                            hint: 'An eclipsing binary usually shows a second dip halfway around the orbit (when the fainter star passes behind the brighter). No secondary eclipse is a point in favour of a planet.',
                            test: res.vetting.secondary_eclipse,
                          },
                        ].map(({ name, hint, test }) => (
                          <div key={name} className={`p-4 rounded-lg border ${
                            test.flag === 'PASS'    ? 'bg-green-50 border-green-100'
                            : test.flag === 'FAIL'  ? 'bg-red-50 border-red-100'
                            : 'bg-yellow-50 border-yellow-100'
                          }`}>
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex-1 min-w-0">
                                <p className="font-semibold text-[#202124] text-sm">{name}</p>
                                <p className="text-xs text-[#5f6368] mt-0.5">{test.message}</p>
                                <p className="text-xs text-[#9aa0a6] mt-1 italic">{hint}</p>
                              </div>
                              <FlagChip flag={test.flag} />
                            </div>
                          </div>
                        ))}
                      </div>
                      <p className="text-xs text-[#5f6368] mt-4 pt-4 border-t border-[#dadce0]">
                        <span className="font-medium">Overall vetting confidence:</span> {(res.vetting.confidence * 100).toFixed(0)}%.
                        {res.vetting.disposition === 'LIKELY_FALSE_POSITIVE' &&
                          ' Follow-up spectroscopy or high-resolution imaging is recommended to confirm the nature of this signal.'}
                        {res.vetting.disposition === 'PLANET_CANDIDATE' &&
                          ' This candidate warrants follow-up radial velocity or transit observations.'}
                        {res.vetting.disposition === 'INCONCLUSIVE' &&
                          ' Additional TESS sectors or ground-based photometry would help clarify the signal.'}
                      </p>
                    </div>
                  )}
                </>
              )}

              {/* No detection message */}
              {!detected && (
                <div className="bg-white border border-[#dadce0] rounded-xl p-6 shadow-sm text-[#5f6368] text-sm">
                  <p className="mb-2">
                    No periodic transit signal above the detection threshold (SNR ≥ 7) was found
                    in the available TESS data for TIC {analysis.tic_id}.
                  </p>
                  <p>
                    This could mean the target has no transiting planet, the planet is not in a favourable
                    orbital geometry, or more TESS sectors are needed.
                  </p>
                  <p className="mt-3 text-xs text-[#9aa0a6]">
                    SNR: {res.snr?.toFixed(2) ?? '—'} · Confidence: {(res.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              )}

              {/* Actions */}
              <div className="flex flex-wrap gap-3 pt-2">
                <Link href="/cloud/analyze"
                  className="px-5 py-2.5 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium rounded-lg transition-colors text-sm">
                  Analyze Another
                </Link>
                <Link href="/dashboard"
                  className="px-5 py-2.5 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors text-sm">
                  Dashboard
                </Link>
                <button onClick={downloadCSV}
                  className="flex items-center gap-2 px-5 py-2.5 bg-white border border-[#dadce0] hover:bg-[#f1f3f4] text-[#202124] font-medium rounded-lg transition-colors text-sm">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  CSV
                </button>
                <button onClick={downloadJSON}
                  className="flex items-center gap-2 px-5 py-2.5 bg-white border border-[#dadce0] hover:bg-[#f1f3f4] text-[#202124] font-medium rounded-lg transition-colors text-sm">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  JSON
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
