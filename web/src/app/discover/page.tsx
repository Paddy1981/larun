'use client'

/**
 * /discover — Citizen Discovery Engine page
 *
 * Step 1: Pick sky region (SkyMap)
 * Step 2: Choose data sources & radius
 * Step 3: Run discovery (POST /api/v2/discover)
 * Step 4: Browse candidates + verify
 */

import { useState } from 'react'
import Link from 'next/link'
import { useSession } from 'next-auth/react'
import Header from '@/components/Header'
import { SkyMap } from '@/components/discovery/SkyMap'
import { CandidateCard } from '@/components/discovery/CandidateCard'
import { ModelResultsPanel } from '@/components/discovery/ModelResultsPanel'
import { DiscoveryReport as DiscoveryReportView } from '@/components/discovery/DiscoveryReport'
import { LightCurveViewer } from '@/components/discovery/LightCurveViewer'
import { VerificationUI } from '@/components/discovery/VerificationUI'
import {
  discoveryClient,
  DiscoveryReport,
  DiscoveryCandidate,
} from '@/lib/discovery-client'
import {
  Telescope, Play, Loader2, AlertCircle, ChevronLeft,
  ChevronRight, Filter, Zap, Search
} from 'lucide-react'

type Step = 'select' | 'configure' | 'results' | 'detail'

const DATA_SOURCES = ['tess', 'kepler', 'neowise']

const SOURCE_META: Record<string, { label: string; description: string }> = {
  tess:    { label: 'TESS',    description: 'NASA transiting exoplanet survey' },
  kepler:  { label: 'Kepler',  description: 'Kepler / K2 photometry archive' },
  neowise: { label: 'NEOWISE', description: 'Wide-field infrared survey (WISE)' },
}

function StepIndicator({ current }: { current: Step }) {
  const steps: { key: Step; label: string }[] = [
    { key: 'select',    label: '1. Sky Region' },
    { key: 'configure', label: '2. Configure' },
    { key: 'results',   label: '3. Results' },
  ]
  const idx = steps.findIndex(s => s.key === current || (current === 'detail' && s.key === 'results'))
  return (
    <div className="flex items-center gap-2">
      {steps.map((s, i) => (
        <div key={s.key} className="flex items-center gap-2">
          <div className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1 rounded-full ${
            i <= idx ? 'bg-[#202124] text-white' : 'bg-[#f1f3f4] text-[#9ca3af]'
          }`}>
            {s.label}
          </div>
          {i < steps.length - 1 && <ChevronRight className="w-3 h-3 text-[#dadce0]" />}
        </div>
      ))}
    </div>
  )
}

export default function DiscoverPage() {
  const { data: session } = useSession()
  const currentUserId = session?.user?.email ?? session?.user?.name ?? 'anonymous'

  // Step state
  const [step, setStep] = useState<Step>('select')

  // Region
  const [ra, setRa] = useState(56.75)
  const [dec, setDec] = useState(24.12)
  const [radiusDeg, setRadiusDeg] = useState(0.5)
  const [selectedSources, setSelectedSources] = useState<string[]>(['tess', 'neowise'])

  // Natural language query
  const [nlQuery, setNlQuery] = useState('')
  const [useNL, setUseNL] = useState(false)

  // Results
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [report, setReport] = useState<DiscoveryReport | null>(null)
  const [activeTab, setActiveTab] = useState<'candidates' | 'anomalies' | 'known' | 'report'>('candidates')

  // Detail view
  const [selectedCandidate, setSelectedCandidate] = useState<DiscoveryCandidate | null>(null)

  // Filter
  const [minPriority, setMinPriority] = useState(0)
  const [onlyNew, setOnlyNew] = useState(false)

  function toggleSource(src: string) {
    setSelectedSources(s => s.includes(src) ? s.filter(x => x !== src) : [...s, src])
  }

  async function runDiscovery() {
    if (selectedSources.length === 0) {
      setError('Select at least one data source')
      return
    }
    setLoading(true)
    setError(null)
    try {
      let result: DiscoveryReport
      if (useNL && nlQuery.trim()) {
        result = await discoveryClient.discoverNL(nlQuery)
      } else {
        result = await discoveryClient.discover({
          ra, dec,
          radius_deg: radiusDeg,
          sources: selectedSources,
          models: 'all',
        })
      }
      setReport(result)
      setStep('results')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Discovery failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  function viewDetail(c: DiscoveryCandidate) {
    setSelectedCandidate(c)
    setStep('detail')
  }

  function backToResults() {
    setSelectedCandidate(null)
    setStep('results')
  }

  // Filtered candidates
  const filteredCandidates = (report?.candidates ?? []).filter(c =>
    c.priority >= minPriority && (!onlyNew || !c.catalog_match.known)
  )

  return (
    <div className="min-h-screen bg-[#f8f9fa]">
      <Header />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-24 pb-16">
        {/* Page header */}
        <div className="mb-8">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Telescope className="w-5 h-5 text-[#1a73e8]" />
                <h1 className="text-2xl font-semibold text-[#202124]">Citizen Discovery Engine</h1>
              </div>
              <p className="text-sm text-[#5f6368]">
                Find new objects in NASA archives using 12 specialized TinyML models — no PhD required.
              </p>
              <p className="text-xs text-[#9ca3af] mt-1">
                Have a specific TESS target or FITS file?{' '}
                <Link href="/cloud/analyze" className="text-[#1a73e8] hover:underline">
                  Open Cloud Analyze →
                </Link>
              </p>
            </div>
            <div className="flex items-center gap-3">
              <StepIndicator current={step} />
              <Link href="/leaderboard" className="btn btn-outline text-sm py-2 px-4">
                Leaderboard
              </Link>
            </div>
          </div>
        </div>

        {/* ── STEP 1: SELECT SKY REGION ── */}
        {step === 'select' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <SkyMap
                onSelect={(r, d) => { setRa(r); setDec(d) }}
                initialRA={ra}
                initialDec={dec}
                radiusDeg={radiusDeg}
                height={480}
              />
            </div>
            <div className="space-y-5">
              {/* Explainer */}
              <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-5">
                <h3 className="font-medium text-[#202124] text-sm mb-3">How it works</h3>
                <ol className="space-y-2 text-sm text-[#5f6368]">
                  {[
                    'Pick any sky region using the interactive map',
                    'Choose your data sources (TESS, Kepler, NEOWISE)',
                    'All 12 TinyML models run automatically in parallel',
                    'Cross-match against 6+ catalogs to find new objects',
                    'Submit candidates for community verification',
                    'Get permanently credited for confirmed discoveries',
                  ].map((step, i) => (
                    <li key={i} className="flex gap-2">
                      <span className="w-5 h-5 rounded-full bg-[#e8f0fe] text-[#1a73e8] text-xs font-medium flex items-center justify-center shrink-0 mt-0.5">
                        {i + 1}
                      </span>
                      {step}
                    </li>
                  ))}
                </ol>
              </div>

              {/* NL query toggle */}
              <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-5">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium text-[#202124] text-sm">Natural language query</h3>
                  <button
                    onClick={() => setUseNL(v => !v)}
                    className={`w-10 h-6 rounded-full transition-colors relative ${useNL ? 'bg-[#1a73e8]' : 'bg-[#dadce0]'}`}
                  >
                    <span className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${useNL ? 'translate-x-4' : 'translate-x-0.5'}`} />
                  </button>
                </div>
                {useNL && (
                  <textarea
                    className="w-full text-sm border border-[#dadce0] rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#1a73e8] resize-none"
                    rows={3}
                    placeholder="e.g. 'Find pulsating stars in the Kepler field with periods between 1-10 days'"
                    value={nlQuery}
                    onChange={e => setNlQuery(e.target.value)}
                  />
                )}
                {!useNL && (
                  <p className="text-xs text-[#9ca3af]">
                    Enable to describe what you're looking for in plain English (powered by Claude)
                  </p>
                )}
              </div>

              <button
                onClick={() => setStep('configure')}
                className="w-full btn btn-primary py-3 flex items-center justify-center gap-2"
              >
                Configure & Run
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* ── STEP 2: CONFIGURE ── */}
        {step === 'configure' && (
          <div className="max-w-2xl mx-auto space-y-5">
            <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-6">
              <h3 className="font-medium text-[#202124] mb-4">Search Configuration</h3>

              {/* Coords summary */}
              <div className="bg-[#f8f9fa] rounded-xl p-4 mb-5 text-sm">
                <div className="flex justify-between">
                  <span className="text-[#5f6368]">RA</span>
                  <span className="font-mono text-[#202124]">{ra.toFixed(4)}°</span>
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-[#5f6368]">Dec</span>
                  <span className="font-mono text-[#202124]">{dec > 0 ? '+' : ''}{dec.toFixed(4)}°</span>
                </div>
              </div>

              {/* Radius */}
              <div className="mb-5">
                <label className="text-sm font-medium text-[#202124] block mb-2">
                  Search radius: {radiusDeg} deg
                </label>
                <input
                  type="range" min={0.1} max={5} step={0.1}
                  value={radiusDeg}
                  onChange={e => setRadiusDeg(+e.target.value)}
                  className="w-full accent-[#1a73e8]"
                />
                <div className="flex justify-between text-xs text-[#9ca3af] mt-1">
                  <span>0.1°</span>
                  <span>5°</span>
                </div>
              </div>

              {/* Data sources */}
              <div className="mb-5">
                <p className="text-sm font-medium text-[#202124] mb-3">Data sources</p>
                <div className="space-y-2">
                  {DATA_SOURCES.map(src => (
                    <label key={src} className="flex items-start gap-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={selectedSources.includes(src)}
                        onChange={() => toggleSource(src)}
                        className="mt-0.5 accent-[#1a73e8]"
                      />
                      <div>
                        <p className="text-sm font-medium text-[#202124]">{SOURCE_META[src].label}</p>
                        <p className="text-xs text-[#9ca3af]">{SOURCE_META[src].description}</p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {error && (
                <div className="flex items-center gap-2 text-sm text-[#dc2626] bg-[#fce8e6] rounded-lg px-4 py-3 mb-4">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  {error}
                </div>
              )}

              <div className="flex gap-3">
                <button
                  onClick={() => setStep('select')}
                  className="btn btn-outline py-2.5 px-5 flex items-center gap-2"
                >
                  <ChevronLeft className="w-4 h-4" />
                  Back
                </button>
                <button
                  onClick={runDiscovery}
                  disabled={loading}
                  className="flex-1 btn btn-primary py-2.5 flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing…</>
                  ) : (
                    <><Play className="w-4 h-4" /> Run Discovery</>
                  )}
                </button>
              </div>
            </div>

            {/* What runs */}
            <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-5">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="w-4 h-4 text-[#1a73e8]" />
                <h3 className="font-medium text-[#202124] text-sm">12 TinyML models will run</h3>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {['VARDET-001','ANOMALY-001','DEBLEND-001','PERIODOGRAM-001',
                  'EXOPLANET-001','VSTAR-001','FLARE-001','ASTERO-001',
                  'SUPERNOVA-001','MICROLENS-001','GALAXY-001','SPECTYPE-001'].map(id => (
                  <span key={id} className="text-xs font-mono px-2 py-0.5 rounded-full bg-[#f1f3f4] text-[#5f6368]">
                    {id}
                  </span>
                ))}
              </div>
              <p className="text-xs text-[#9ca3af] mt-3">
                All CPU-only · &lt;500ms per object · No GPU required
              </p>
            </div>
          </div>
        )}

        {/* ── STEP 3: RESULTS ── */}
        {step === 'results' && report && (
          <div className="space-y-5">
            {/* Stats bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: 'Analyzed',   value: report.stats?.analyzed   ?? 0,                                              color: '#202124' },
                { label: 'Candidates', value: report.stats?.candidates ?? 0,                                              color: '#1a73e8' },
                { label: 'Known',      value: report.stats?.known      ?? 0,                                              color: '#5f6368' },
                { label: 'Time',       value: `${(report.stats?.elapsed_seconds ?? 0).toFixed(1)}s`,                      color: '#16a34a' },
              ].map(s => (
                <div key={s.label} className="bg-white rounded-xl border border-[#e5e7eb] shadow-sm px-5 py-4 text-center">
                  <p className="text-2xl font-semibold" style={{ color: s.color }}>{s.value}</p>
                  <p className="text-xs text-[#9ca3af] mt-0.5">{s.label}</p>
                </div>
              ))}
            </div>

            {/* Tabs */}
            <div className="flex items-center gap-1 bg-white rounded-xl border border-[#e5e7eb] shadow-sm p-1 w-fit">
              {([
                ['candidates', `Candidates (${(report.candidates ?? []).length})`],
                ['anomalies',  `Anomalies (${(report.anomalies ?? []).length})`],
                ['known',      `Known (${(report.known ?? []).length})`],
                ['report',     'Full Report'],
              ] as const).map(([tab, label]) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`text-sm px-4 py-2 rounded-lg transition-colors ${
                    activeTab === tab
                      ? 'bg-[#202124] text-white'
                      : 'text-[#5f6368] hover:bg-[#f1f3f4]'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Filters (candidates only) */}
            {activeTab === 'candidates' && (
              <div className="flex items-center gap-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <Filter className="w-3.5 h-3.5 text-[#5f6368]" />
                  <span className="text-xs text-[#5f6368]">Min priority:</span>
                  <select
                    value={minPriority}
                    onChange={e => setMinPriority(+e.target.value)}
                    className="text-xs border border-[#dadce0] rounded-lg px-2 py-1 focus:outline-none focus:ring-1 focus:ring-[#1a73e8]"
                  >
                    <option value={0}>All</option>
                    <option value={40}>Low+ (40)</option>
                    <option value={60}>Medium+ (60)</option>
                    <option value={80}>High (80+)</option>
                  </select>
                </div>
                <label className="flex items-center gap-2 text-xs text-[#5f6368] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={onlyNew}
                    onChange={e => setOnlyNew(e.target.checked)}
                    className="accent-[#1a73e8]"
                  />
                  Previously unknown only
                </label>
                <span className="text-xs text-[#9ca3af]">{filteredCandidates.length} shown</span>
              </div>
            )}

            {/* Candidates grid */}
            {activeTab === 'candidates' && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredCandidates.length === 0 ? (
                  <div className="col-span-full bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-12 text-center">
                    <Search className="w-8 h-8 text-[#dadce0] mx-auto mb-3" />
                    <p className="text-sm text-[#5f6368]">No candidates match the current filters</p>
                  </div>
                ) : (
                  filteredCandidates.map((c, i) => (
                    <CandidateCard
                      key={i}
                      candidate={c}
                      rank={i + 1}
                      onViewDetails={viewDetail}
                    />
                  ))
                )}
              </div>
            )}

            {/* Anomalies grid */}
            {activeTab === 'anomalies' && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {(report.anomalies ?? []).length === 0 ? (
                  <div className="col-span-full bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-12 text-center">
                    <p className="text-sm text-[#5f6368]">No anomalies detected in this region</p>
                  </div>
                ) : (
                  (report.anomalies ?? []).map((c, i) => (
                    <CandidateCard key={i} candidate={c} rank={i + 1} onViewDetails={viewDetail} />
                  ))
                )}
              </div>
            )}

            {/* Known objects list */}
            {activeTab === 'known' && (
              <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm divide-y divide-[#f1f3f4]">
                {(report.known ?? []).length === 0 ? (
                  <div className="p-12 text-center text-sm text-[#5f6368]">No known objects found</div>
                ) : (
                  (report.known ?? []).slice(0, 50).map((c, i) => (
                    <div key={i} className="px-5 py-3 flex items-center justify-between hover:bg-[#fafafa]">
                      <div>
                        <p className="text-sm font-medium text-[#202124]">
                          {c.catalog_match.matches?.[0]?.name ?? `Object ${i + 1}`}
                        </p>
                        <p className="text-xs text-[#9ca3af] font-mono">
                          {c.target.ra.toFixed(4)}° / {c.target.dec.toFixed(4)}°
                        </p>
                      </div>
                      <span className="text-xs text-[#5f6368]">
                        {c.catalog_match.matches?.[0]?.catalog ?? 'catalog'}
                      </span>
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Full report */}
            {activeTab === 'report' && (
              <DiscoveryReportView report={report} />
            )}

            {/* New search button */}
            <div className="flex justify-center">
              <button
                onClick={() => { setReport(null); setStep('select') }}
                className="btn btn-outline px-6 py-2.5 flex items-center gap-2"
              >
                <ChevronLeft className="w-4 h-4" />
                New search
              </button>
            </div>
          </div>
        )}

        {/* ── STEP 4: DETAIL VIEW ── */}
        {step === 'detail' && selectedCandidate && (
          <div className="space-y-5">
            <button
              onClick={backToResults}
              className="flex items-center gap-2 text-sm text-[#1a73e8] hover:underline"
            >
              <ChevronLeft className="w-4 h-4" />
              Back to results
            </button>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              {/* Left: light curve + verification */}
              <div className="lg:col-span-2 space-y-5">
                <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-5">
                  <h2 className="font-medium text-[#202124] mb-1">
                    {selectedCandidate.consensus.consensus_label.replace(/_/g, ' ')}
                  </h2>
                  <p className="text-xs text-[#9ca3af] font-mono mb-4">
                    RA {selectedCandidate.target.ra.toFixed(4)}° / Dec {selectedCandidate.target.dec.toFixed(4)}°
                  </p>
                  <LightCurveViewer
                    times={[]}
                    flux={[]}
                    period={selectedCandidate.period_days}
                    title="Light Curve (load from pipeline)"
                    height={280}
                  />
                </div>

                <VerificationUI
                  candidate={selectedCandidate}
                  discoveryId={`disc-${selectedCandidate.target.ra}-${selectedCandidate.target.dec}`}
                  currentUserId={currentUserId}
                />
              </div>

              {/* Right: model results */}
              <div>
                <ModelResultsPanel
                  results={selectedCandidate.classifications}
                  consensus={selectedCandidate.consensus}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
