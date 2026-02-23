'use client'

import { useState, useEffect, useRef } from 'react'
import { useSession } from 'next-auth/react'
import { FileUpload } from '@/components/FileUpload'
import { ModelSelector } from '@/components/ModelSelector'
import { ResultsDisplay } from '@/components/ResultsDisplay'
import { ColorIndexInput, type ColorIndices } from '@/components/ColorIndexInput'
import { getCurrentUser, getUserQuota, type UsageQuota } from '@/lib/supabase'
import { apiClient, getModelById } from '@/lib/api-client'
import type { InferenceResult } from '@/lib/supabase'
import { Loader2, AlertCircle, Search, Upload, Telescope, Zap, CheckCircle2, XCircle, AlertTriangle, RotateCcw, ChevronRight } from 'lucide-react'
import Link from 'next/link'

// ── Types ─────────────────────────────────────────────────────────────────────

interface VettingTest { flag: 'PASS' | 'FAIL' | 'WARNING'; message: string }
interface TicResult {
  detection: boolean
  confidence: number
  period_days: number | null
  depth_ppm: number | null
  duration_hours: number | null
  vetting?: {
    disposition: string
    odd_even: VettingTest
    v_shape: VettingTest
    secondary_eclipse: VettingTest
  }
}

const POPULAR_TARGETS = [
  { id: '470710327', name: 'TOI-1338 b', type: 'Circumbinary',    year: '2020' },
  { id: '307210830', name: 'TOI-700 d',  type: 'Earth-sized HZ',  year: '2020' },
  { id: '261136679', name: 'TOI-175 b',  type: 'Super-Earth',     year: '2019' },
]

const PIPELINE_STAGES = [
  { label: 'Fetching TESS data',       sub: 'NASA MAST archive' },
  { label: 'Preprocessing light curve', sub: 'Normalise & detrend' },
  { label: 'Running BLS periodogram',  sub: 'Transit search' },
  { label: 'Vetting candidate',         sub: 'False positive tests' },
]

const EMPTY_INDICES: ColorIndices = { bv: '', vr: '', bp_rp: '', jh: '', hk: '' }

// ── Component ─────────────────────────────────────────────────────────────────

export default function AnalyzePage() {
  const { data: nextAuthSession, status: nextAuthStatus } = useSession()
  const autoTicRef = useRef(false)

  const [supabaseUser, setSupabaseUser] = useState<any>(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [quota, setQuota] = useState<UsageQuota | null>(null)

  const [mode, setMode] = useState<'tic' | 'upload'>('tic')

  // TIC ID state
  const [ticId, setTicId]         = useState('')
  const [ticLoading, setTicLoading]   = useState(false)
  const [ticProgress, setTicProgress] = useState(0)
  const [ticStage, setTicStage]     = useState(0)
  const [ticResult, setTicResult]   = useState<TicResult | null>(null)
  const [ticError, setTicError]     = useState<string | null>(null)
  const [ticId_used, setTicIdUsed]  = useState('')

  // Past analyses history
  const [pastAnalyses, setPastAnalyses] = useState<Array<{
    id: string
    tic_id: string
    status: string
    created_at: string
    result?: TicResult
  }>>([])
  const [historyLoading, setHistoryLoading] = useState(false)

  // Upload state
  const [selectedModel, setSelectedModel]   = useState('EXOPLANET-001')
  const [file, setFile]                     = useState<File | null>(null)
  const [colorIndices, setColorIndices]     = useState<ColorIndices>(EMPTY_INDICES)
  const [uploadResult, setUploadResult]     = useState<InferenceResult | null>(null)
  const [uploadLoading, setUploadLoading]   = useState(false)
  const [uploadError, setUploadError]       = useState<string | null>(null)

  // Resolved user — Supabase first, NextAuth fallback
  const user = supabaseUser ?? (nextAuthSession?.user ? {
    id: nextAuthSession.user.id ?? nextAuthSession.user.email,
    email: nextAuthSession.user.email,
    name: nextAuthSession.user.name,
    _source: 'nextauth',
  } : null)

  useEffect(() => {
    if (nextAuthStatus === 'loading') return
    checkAuth()
  }, [nextAuthStatus])

  const checkAuth = async () => {
    const { user: sbUser } = await getCurrentUser()
    setSupabaseUser(sbUser)
    if (sbUser) setQuota(await getUserQuota(sbUser.id))
    setAuthLoading(false)
  }

  // Pre-fill TIC ID from ?tic= query param and auto-trigger once auth resolves
  useEffect(() => {
    if (authLoading || autoTicRef.current) return
    const tic = new URLSearchParams(window.location.search).get('tic')
    if (!tic) return
    autoTicRef.current = true
    setMode('tic')
    setTicId(tic)
    // Small delay so the input renders before analysis fires
    setTimeout(() => handleTicAnalyze(tic), 300)
  }, [authLoading])

  const isQuotaExceeded =
    quota !== null && quota.quota_limit !== null &&
    quota.quota_limit !== -1 && quota.analyses_count >= quota.quota_limit

  const refreshQuota = async () => {
    if (supabaseUser) setQuota(await getUserQuota(supabaseUser.id))
  }

  const fetchHistory = async () => {
    setHistoryLoading(true)
    try {
      const res = await fetch('/api/v1/analyses')
      if (res.ok) {
        const d = await res.json()
        setPastAnalyses(
          (d.analyses || [])
            .filter((a: any) => a.tic_id && a.status === 'completed')
            .slice(0, 30)
        )
      }
    } catch { /* silent */ } finally {
      setHistoryLoading(false)
    }
  }

  useEffect(() => {
    if (!authLoading && user) fetchHistory()
  }, [authLoading])

  const loadPastResult = (a: typeof pastAnalyses[0]) => {
    setMode('tic')
    setTicId(a.tic_id)
    setTicIdUsed(a.tic_id)
    setTicResult(a.result ?? null)
    setTicError(null)
  }

  // ── TIC ID analysis ──────────────────────────────────────────────────────────
  const handleTicAnalyze = async (targetId?: string) => {
    const id = (targetId ?? ticId).trim()
    if (!id || !user) return

    setTicLoading(true); setTicError(null); setTicResult(null)
    setTicProgress(0);   setTicStage(0);   setTicIdUsed(id)

    const tick = setInterval(() => {
      setTicProgress(p => {
        const next = Math.min(92, p + 4)
        setTicStage(Math.floor((next / 100) * PIPELINE_STAGES.length))
        return next
      })
    }, 1200)

    try {
      const res = await fetch('/api/v1/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tic_id: id,
          ...(supabaseUser ? { user_id: supabaseUser.id } : {}),
        }),
      })
      clearInterval(tick); setTicProgress(100); setTicStage(PIPELINE_STAGES.length)

      const text = await res.text()
      let data: any
      try { data = JSON.parse(text) } catch {
        throw new Error(res.status === 504
          ? 'Analysis timed out. TESS data can take up to 60 s — please try again.'
          : `Server error (${res.status}). Please try again.`)
      }
      if (!res.ok) throw new Error(data.error?.message || 'Analysis failed')
      if (data.status === 'failed') throw new Error(data.error || 'Detection failed')

      setTicResult(data.result)
      await refreshQuota()
      fetchHistory()
    } catch (err: any) {
      setTicError(err.message || 'An error occurred')
    } finally {
      clearInterval(tick); setTicLoading(false)
    }
  }

  // ── Upload analysis ──────────────────────────────────────────────────────────
  const isSpectral   = selectedModel === 'SPECTYPE-001'
  const hasColorIndex = Object.values(colorIndices).some(v => v.trim() !== '' && !isNaN(Number(v)))
  const canRunUpload  = isSpectral ? hasColorIndex : !!file

  const handleUploadAnalyze = async () => {
    if (!user || !canRunUpload || isQuotaExceeded) return
    setUploadLoading(true); setUploadError(null); setUploadResult(null)
    try {
      const uid = supabaseUser?.id ?? user.id
      let res: InferenceResult
      if (isSpectral) {
        const parse = (v: string) => v.trim() === '' ? undefined : Number(v)
        res = await apiClient.analyzeSpectralType(
          { bv: parse(colorIndices.bv), vr: parse(colorIndices.vr),
            bp_rp: parse(colorIndices.bp_rp), jh: parse(colorIndices.jh), hk: parse(colorIndices.hk) },
          uid
        )
      } else {
        res = await apiClient.analyzeTinyML(file!, selectedModel, uid)
      }
      setUploadResult(res)
      await refreshQuota()
    } catch (err: any) {
      setUploadError(err.response?.data?.detail || err.message || 'Analysis failed. Please try again.')
    } finally {
      setUploadLoading(false)
    }
  }

  // ── Loading ──────────────────────────────────────────────────────────────────
  if (authLoading || nextAuthStatus === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#fafafa]">
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-[#e5e7eb] border-t-[#202124] rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-[#6b7280]">Loading…</p>
        </div>
      </div>
    )
  }

  // ── Not logged in ────────────────────────────────────────────────────────────
  if (!user) {
    return (
      <div className="min-h-screen bg-[#fafafa] flex items-center justify-center px-6">
        <div className="max-w-lg w-full">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-[#202124] rounded-2xl flex items-center justify-center mx-auto mb-5">
              <Telescope className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-semibold text-[#202124] mb-2">Analyze Astronomical Data</h1>
            <p className="text-[#6b7280]">
              Sign in to run TIC ID lookups and upload FITS files.
              5 free analyses every month — no credit card required.
            </p>
          </div>
          <div className="bg-white rounded-2xl border border-[#e5e7eb] p-6 shadow-sm">
            <div className="grid grid-cols-3 gap-4 mb-6 pb-6 border-b border-[#f3f4f6]">
              {[
                { icon: '🔭', label: 'TESS Lookup',    sub: 'BLS transit detection' },
                { icon: '📁', label: 'File Upload',    sub: '8 TinyML models' },
                { icon: '⚡', label: 'Instant results', sub: '<10 ms inference' },
              ].map(f => (
                <div key={f.label} className="text-center">
                  <div className="text-2xl mb-1">{f.icon}</div>
                  <p className="text-xs font-medium text-[#202124]">{f.label}</p>
                  <p className="text-xs text-[#9ca3af]">{f.sub}</p>
                </div>
              ))}
            </div>
            <div className="flex flex-col gap-3">
              <Link href="/cloud/auth/login?redirect=/cloud/analyze"
                className="flex items-center justify-center gap-2 w-full py-3 bg-[#202124] hover:bg-[#374151] text-white text-sm font-medium rounded-xl transition-colors">
                Sign In
                <ChevronRight className="w-4 h-4" />
              </Link>
              <Link href="/cloud/auth/signup?redirect=/cloud/analyze"
                className="flex items-center justify-center gap-2 w-full py-3 bg-white hover:bg-[#f9fafb] text-[#202124] text-sm font-medium rounded-xl border border-[#e5e7eb] transition-colors">
                Create Free Account
              </Link>
            </div>
          </div>
        </div>
      </div>
    )
  }

  const selectedModelMeta = getModelById(selectedModel)

  return (
    <div className="min-h-screen bg-[#fafafa]">

      {/* ── Page Header ──────────────────────────────────────────────────────── */}
      <div className="bg-white border-b border-[#e5e7eb]">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <h1 className="text-2xl font-semibold text-[#202124] mb-1">Analyze</h1>
              <p className="text-sm text-[#6b7280]">
                Look up a TESS target by ID, or upload your own data file for TinyML inference.
              </p>
            </div>

            {/* Quota badge */}
            {quota && (
              <div className={`flex items-center gap-3 px-4 py-2.5 rounded-xl border text-sm ${
                isQuotaExceeded
                  ? 'bg-red-50 border-red-200 text-red-700'
                  : quota.analyses_count >= (quota.quota_limit ?? 5) * 0.8
                  ? 'bg-amber-50 border-amber-200 text-amber-700'
                  : 'bg-[#f0fdf4] border-[#bbf7d0] text-[#15803d]'
              }`}>
                <div className="text-right">
                  <p className="font-semibold text-base leading-none">
                    {quota.analyses_count} <span className="font-normal text-xs">/ {quota.quota_limit === -1 ? '∞' : quota.quota_limit}</span>
                  </p>
                  <p className="text-xs opacity-70 mt-0.5">analyses this month</p>
                </div>
                {isQuotaExceeded && (
                  <Link href="/cloud/pricing" className="text-xs font-medium underline whitespace-nowrap">
                    Upgrade →
                  </Link>
                )}
              </div>
            )}
          </div>

          {/* Mode tabs */}
          <div className="flex gap-1 mt-6 border-b border-[#e5e7eb] -mb-px">
            {([
              { key: 'tic',    label: 'TIC ID Lookup', icon: Search, sub: 'TESS target by ID' },
              { key: 'upload', label: 'Upload File',    icon: Upload, sub: 'FITS / photometry' },
            ] as const).map(tab => (
              <button
                key={tab.key}
                onClick={() => {
                  setMode(tab.key)
                  if (tab.key === 'tic') { setTicResult(null); setTicError(null) }
                  else { setUploadResult(null); setUploadError(null) }
                }}
                className={`flex items-center gap-2.5 px-5 py-3 text-sm font-medium border-b-2 transition-colors ${
                  mode === tab.key
                    ? 'border-[#202124] text-[#202124]'
                    : 'border-transparent text-[#6b7280] hover:text-[#374151]'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
                <span className={`text-xs hidden sm:inline ${mode === tab.key ? 'text-[#9ca3af]' : 'text-[#d1d5db]'}`}>
                  {tab.sub}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Quota exceeded banner ─────────────────────────────────────────────── */}
      {isQuotaExceeded && (
        <div className="max-w-6xl mx-auto px-6 pt-6">
          <div className="flex items-center gap-4 bg-amber-50 border border-amber-200 rounded-xl px-5 py-4">
            <AlertCircle className="w-5 h-5 text-amber-500 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-amber-900">Monthly quota reached</p>
              <p className="text-xs text-amber-700">You have used all {quota?.quota_limit} analyses this month.</p>
            </div>
            <Link href="/cloud/pricing" className="text-xs font-semibold text-amber-900 bg-amber-100 hover:bg-amber-200 px-3 py-1.5 rounded-lg transition-colors whitespace-nowrap">
              View Plans →
            </Link>
          </div>
        </div>
      )}

      <div className="max-w-6xl mx-auto px-6 py-8">

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* TIC ID MODE                                                        */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        {mode === 'tic' && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

            {/* Left column — input */}
            <div className="lg:col-span-2 space-y-4">

              {/* Search card */}
              <div className="bg-white rounded-2xl border border-[#e5e7eb] p-6 shadow-sm">
                <div className="flex items-center gap-2 mb-1">
                  <Telescope className="w-4 h-4 text-[#6b7280]" />
                  <h2 className="text-sm font-semibold text-[#374151] uppercase tracking-wide">TESS Target Lookup</h2>
                </div>
                <p className="text-xs text-[#9ca3af] mb-5">
                  Enter a TIC ID — we fetch the light curve from NASA MAST and run BLS transit detection.
                </p>

                <div className="relative mb-4">
                  <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#9ca3af]" />
                  <input
                    type="text"
                    placeholder="TIC ID — e.g. 470710327"
                    value={ticId}
                    onChange={e => setTicId(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && !ticLoading && ticId.trim() && handleTicAnalyze()}
                    disabled={ticLoading || isQuotaExceeded}
                    className="w-full pl-10 pr-4 py-3 text-sm border border-[#e5e7eb] rounded-xl focus:outline-none focus:ring-2 focus:ring-[#202124] focus:border-transparent font-mono bg-[#fafafa] disabled:opacity-50 transition"
                  />
                </div>

                <button
                  onClick={() => handleTicAnalyze()}
                  disabled={!ticId.trim() || ticLoading || isQuotaExceeded}
                  className="w-full flex items-center justify-center gap-2 py-3 bg-[#202124] hover:bg-[#374151] disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-xl transition-colors"
                >
                  {ticLoading
                    ? <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing…</>
                    : <><Zap className="w-4 h-4" /> Run Analysis</>}
                </button>

                {ticError && (
                  <div className="mt-4 flex items-start gap-3 bg-red-50 border border-red-100 rounded-xl p-4">
                    <AlertCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                    <p className="text-xs text-red-700">{ticError}</p>
                  </div>
                )}
              </div>

              {/* Popular targets */}
              {!ticLoading && !ticResult && (
                <div className="bg-white rounded-2xl border border-[#e5e7eb] p-5 shadow-sm">
                  <p className="text-xs font-semibold text-[#9ca3af] uppercase tracking-wide mb-3">Known Targets</p>
                  <div className="space-y-2">
                    {POPULAR_TARGETS.map(t => (
                      <button
                        key={t.id}
                        onClick={() => { setTicId(t.id); handleTicAnalyze(t.id) }}
                        disabled={isQuotaExceeded || ticLoading}
                        className="w-full flex items-center justify-between px-4 py-3 rounded-xl hover:bg-[#f3f4f6] border border-transparent hover:border-[#e5e7eb] transition-all disabled:opacity-50 text-left"
                      >
                        <div>
                          <p className="text-sm font-medium text-[#202124]">{t.name}</p>
                          <p className="text-xs text-[#9ca3af]">{t.type} · {t.year}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-xs font-mono text-[#9ca3af]">{t.id}</p>
                          <ChevronRight className="w-3.5 h-3.5 text-[#d1d5db] ml-auto mt-0.5" />
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Past analyses */}
              {!ticLoading && pastAnalyses.length > 0 && (
                <div className="bg-white rounded-2xl border border-[#e5e7eb] p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-xs font-semibold text-[#9ca3af] uppercase tracking-wide">History</p>
                    <button onClick={fetchHistory} disabled={historyLoading}
                      className="text-xs text-[#9ca3af] hover:text-[#6b7280] transition-colors disabled:opacity-40">
                      {historyLoading ? '…' : 'Refresh'}
                    </button>
                  </div>
                  <div className="space-y-1.5 max-h-64 overflow-y-auto">
                    {pastAnalyses.map(a => {
                      const detected = a.result?.detection
                      const conf = a.result?.confidence != null ? `${(a.result.confidence * 100).toFixed(0)}%` : null
                      const age = (() => {
                        const d = Date.now() - new Date(a.created_at).getTime()
                        const m = Math.floor(d / 60000)
                        if (m < 60) return `${m}m ago`
                        const h = Math.floor(m / 60)
                        if (h < 24) return `${h}h ago`
                        return `${Math.floor(h / 24)}d ago`
                      })()
                      const isActive = ticId_used === a.tic_id && ticResult != null
                      return (
                        <button
                          key={a.id}
                          onClick={() => loadPastResult(a)}
                          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-left transition-all ${
                            isActive
                              ? 'bg-[#202124] border border-[#202124]'
                              : 'hover:bg-[#f9fafb] border border-transparent hover:border-[#e5e7eb]'
                          }`}
                        >
                          <div className="min-w-0">
                            <p className={`text-xs font-mono font-medium truncate ${isActive ? 'text-white' : 'text-[#374151]'}`}>
                              TIC {a.tic_id}
                            </p>
                            <p className={`text-xs mt-0.5 ${isActive ? 'text-[#9ca3af]' : 'text-[#9ca3af]'}`}>{age}</p>
                          </div>
                          <div className="shrink-0 ml-2 text-right">
                            {detected != null && (
                              <span className={`text-xs px-1.5 py-0.5 rounded-full font-medium ${
                                detected
                                  ? isActive ? 'bg-green-900 text-green-300' : 'bg-[#dcfce7] text-[#166534]'
                                  : isActive ? 'bg-gray-700 text-gray-300'  : 'bg-[#f3f4f6] text-[#6b7280]'
                              }`}>
                                {detected ? (conf ?? '✓') : 'None'}
                              </span>
                            )}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Pipeline info */}
              {!ticLoading && !ticResult && (
                <div className="bg-white rounded-2xl border border-[#e5e7eb] p-5 shadow-sm">
                  <p className="text-xs font-semibold text-[#9ca3af] uppercase tracking-wide mb-4">How It Works</p>
                  <div className="space-y-3">
                    {PIPELINE_STAGES.map((s, i) => (
                      <div key={s.label} className="flex items-start gap-3">
                        <div className="w-6 h-6 rounded-full bg-[#f3f4f6] flex items-center justify-center text-xs font-bold text-[#6b7280] shrink-0 mt-0.5">
                          {i + 1}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-[#374151]">{s.label}</p>
                          <p className="text-xs text-[#9ca3af]">{s.sub}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Right column — results */}
            <div className="lg:col-span-3">
              {ticLoading ? (
                <TicLoadingCard ticId={ticId_used} progress={ticProgress} stage={ticStage} />
              ) : ticResult ? (
                <TicResultCard result={ticResult} ticId={ticId_used} onReset={() => { setTicResult(null); setTicId('') }} />
              ) : (
                <div className="bg-white rounded-2xl border border-[#e5e7eb] h-full min-h-[420px] flex items-center justify-center shadow-sm">
                  <div className="text-center p-8">
                    <div className="w-16 h-16 bg-[#f3f4f6] rounded-2xl flex items-center justify-center mx-auto mb-4">
                      <Telescope className="w-8 h-8 text-[#d1d5db]" />
                    </div>
                    <p className="text-sm font-medium text-[#374151] mb-1">Results will appear here</p>
                    <p className="text-xs text-[#9ca3af]">Enter a TIC ID or pick a known target</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════════════ */}
        {/* UPLOAD MODE                                                        */}
        {/* ═══════════════════════════════════════════════════════════════════ */}
        {mode === 'upload' && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

            {/* Left column — configuration */}
            <div className="lg:col-span-2 space-y-4">

              {/* Step 1 — Model */}
              <div className="bg-white rounded-2xl border border-[#e5e7eb] p-6 shadow-sm">
                <div className="flex items-center gap-2 mb-4">
                  <span className="w-6 h-6 rounded-full bg-[#202124] text-white text-xs font-bold flex items-center justify-center shrink-0">1</span>
                  <h2 className="text-sm font-semibold text-[#374151]">Choose a Model</h2>
                </div>
                <ModelSelector
                  selectedModel={selectedModel}
                  onModelSelect={id => { setSelectedModel(id); setUploadResult(null); setUploadError(null) }}
                  disabled={uploadLoading}
                />
              </div>

              {/* Step 2 — Data input */}
              <div className="bg-white rounded-2xl border border-[#e5e7eb] p-6 shadow-sm">
                <div className="flex items-center gap-2 mb-4">
                  <span className="w-6 h-6 rounded-full bg-[#202124] text-white text-xs font-bold flex items-center justify-center shrink-0">2</span>
                  <h2 className="text-sm font-semibold text-[#374151]">
                    {isSpectral ? 'Enter Colour Indices' : 'Upload Data File'}
                  </h2>
                </div>
                {isSpectral ? (
                  <ColorIndexInput
                    values={colorIndices}
                    onChange={setColorIndices}
                    disabled={uploadLoading || isQuotaExceeded}
                  />
                ) : (
                  <>
                    {selectedModelMeta && (
                      <p className="text-xs text-[#9ca3af] mb-4">
                        <span className="font-medium text-[#6b7280]">Expected:</span> {selectedModelMeta.data_source}
                      </p>
                    )}
                    <FileUpload
                      onFileSelect={setFile}
                      selectedFile={file}
                      disabled={uploadLoading || isQuotaExceeded}
                    />
                  </>
                )}
              </div>

              {/* Step 3 — Run */}
              <div className="bg-white rounded-2xl border border-[#e5e7eb] p-6 shadow-sm">
                <div className="flex items-center gap-2 mb-4">
                  <span className={`w-6 h-6 rounded-full text-white text-xs font-bold flex items-center justify-center shrink-0 ${canRunUpload && !isQuotaExceeded ? 'bg-[#202124]' : 'bg-[#d1d5db]'}`}>3</span>
                  <h2 className="text-sm font-semibold text-[#374151]">Run Analysis</h2>
                </div>
                <button
                  onClick={handleUploadAnalyze}
                  disabled={!canRunUpload || uploadLoading || isQuotaExceeded}
                  className="w-full flex items-center justify-center gap-2 py-3 bg-[#202124] hover:bg-[#374151] disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-xl transition-colors"
                >
                  {uploadLoading
                    ? <><Loader2 className="w-4 h-4 animate-spin" /> Running inference…</>
                    : <><Zap className="w-4 h-4" /> Run Inference</>}
                </button>

                {!canRunUpload && !uploadLoading && (
                  <p className="text-xs text-[#9ca3af] text-center mt-3">
                    {isSpectral ? 'Enter at least one colour index above' : 'Upload a file to continue'}
                  </p>
                )}

                {uploadError && (
                  <div className="mt-4 flex items-start gap-3 bg-red-50 border border-red-100 rounded-xl p-4">
                    <AlertCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-medium text-red-800 mb-0.5">Analysis failed</p>
                      <p className="text-xs text-red-700">{uploadError}</p>
                      {uploadError.includes('quota') && (
                        <Link href="/cloud/pricing" className="text-xs text-red-800 underline mt-1 inline-block">View Plans →</Link>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right column — results */}
            <div className="lg:col-span-3">
              {uploadLoading ? (
                <div className="bg-white rounded-2xl border border-[#e5e7eb] h-full min-h-[420px] flex items-center justify-center shadow-sm">
                  <div className="text-center p-8">
                    <div className="w-14 h-14 relative mx-auto mb-5">
                      <div className="absolute inset-0 border-4 border-[#f3f4f6] rounded-full" />
                      <div className="absolute inset-0 border-4 border-[#202124] rounded-full border-t-transparent animate-spin" />
                    </div>
                    <p className="text-sm font-medium text-[#374151] mb-1">Running inference…</p>
                    <p className="text-xs text-[#9ca3af]">{getModelById(selectedModel)?.name ?? selectedModel}</p>
                  </div>
                </div>
              ) : uploadResult ? (
                <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm overflow-hidden">
                  <ResultsDisplay result={uploadResult} modelId={selectedModel} />
                </div>
              ) : (
                <div className="bg-white rounded-2xl border border-[#e5e7eb] h-full min-h-[420px] flex items-center justify-center shadow-sm">
                  <div className="text-center p-8">
                    <div className="w-16 h-16 bg-[#f3f4f6] rounded-2xl flex items-center justify-center mx-auto mb-4">
                      <Upload className="w-8 h-8 text-[#d1d5db]" />
                    </div>
                    <p className="text-sm font-medium text-[#374151] mb-1">Results will appear here</p>
                    <p className="text-xs text-[#9ca3af]">
                      {isSpectral ? 'Enter colour indices and run analysis' : 'Upload a file and run inference'}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── TIC Loading Card ───────────────────────────────────────────────────────────

function TicLoadingCard({ ticId, progress, stage }: { ticId: string; progress: number; stage: number }) {
  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] p-8 shadow-sm h-full min-h-[420px] flex flex-col justify-center">
      <div className="max-w-sm mx-auto w-full">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-8 h-8 relative shrink-0">
            <div className="absolute inset-0 border-[3px] border-[#f3f4f6] rounded-full" />
            <div className="absolute inset-0 border-[3px] border-[#202124] rounded-full border-t-transparent animate-spin" />
          </div>
          <div>
            <p className="text-sm font-semibold text-[#202124]">Analyzing TIC {ticId}</p>
            <p className="text-xs text-[#9ca3af]">This may take 20–60 seconds</p>
          </div>
        </div>

        {/* Progress bar */}
        <div className="w-full bg-[#f3f4f6] rounded-full h-1.5 mb-6 overflow-hidden">
          <div
            className="bg-[#202124] h-1.5 rounded-full transition-all duration-700 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Pipeline stages */}
        <div className="space-y-3">
          {PIPELINE_STAGES.map((s, i) => {
            const done    = i < stage
            const active  = i === stage
            return (
              <div key={s.label} className={`flex items-center gap-3 transition-opacity ${active ? 'opacity-100' : done ? 'opacity-60' : 'opacity-25'}`}>
                <div className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${
                  done ? 'bg-[#f0fdf4]' : active ? 'bg-[#f3f4f6]' : 'bg-[#f9fafb]'
                }`}>
                  {done
                    ? <CheckCircle2 className="w-4 h-4 text-[#16a34a]" />
                    : active
                    ? <Loader2 className="w-3 h-3 text-[#6b7280] animate-spin" />
                    : <div className="w-1.5 h-1.5 rounded-full bg-[#d1d5db]" />}
                </div>
                <div>
                  <p className="text-sm font-medium text-[#374151]">{s.label}</p>
                  <p className="text-xs text-[#9ca3af]">{s.sub}</p>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// ── Vetting Row ────────────────────────────────────────────────────────────────

function VettingRow({ label, test }: { label: string; test: VettingTest }) {
  const cfg = {
    PASS:    { icon: CheckCircle2,   cls: 'text-[#16a34a]', bg: 'bg-[#f0fdf4]', badge: 'bg-[#dcfce7] text-[#166534]' },
    FAIL:    { icon: XCircle,        cls: 'text-[#dc2626]', bg: 'bg-[#fef2f2]', badge: 'bg-[#fee2e2] text-[#991b1b]' },
    WARNING: { icon: AlertTriangle,  cls: 'text-[#d97706]', bg: 'bg-[#fffbeb]', badge: 'bg-[#fef3c7] text-[#92400e]' },
  }[test.flag]
  const Icon = cfg.icon
  return (
    <div className={`flex items-center gap-3 p-3 rounded-xl ${cfg.bg}`}>
      <Icon className={`w-4 h-4 shrink-0 ${cfg.cls}`} />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-[#374151]">{label}</p>
        <p className="text-xs text-[#6b7280] truncate">{test.message}</p>
      </div>
      <span className={`text-xs px-2 py-0.5 rounded-full font-semibold shrink-0 ${cfg.badge}`}>
        {test.flag}
      </span>
    </div>
  )
}

// ── TIC Result Card ────────────────────────────────────────────────────────────

function TicResultCard({ result, ticId, onReset }: { result: TicResult; ticId: string; onReset: () => void }) {
  const detected = result.detection
  const conf = (result.confidence * 100).toFixed(1)

  return (
    <div className="space-y-4">
      {/* Detection hero */}
      <div className={`rounded-2xl border p-6 shadow-sm ${
        detected ? 'bg-[#f0fdf4] border-[#bbf7d0]' : 'bg-white border-[#e5e7eb]'
      }`}>
        <div className="flex items-start gap-4 mb-5">
          <div className={`w-12 h-12 rounded-2xl flex items-center justify-center shrink-0 ${
            detected ? 'bg-[#dcfce7]' : 'bg-[#f3f4f6]'
          }`}>
            {detected
              ? <CheckCircle2 className="w-7 h-7 text-[#16a34a]" />
              : <XCircle      className="w-7 h-7 text-[#9ca3af]" />}
          </div>
          <div className="flex-1">
            <h3 className={`text-xl font-semibold ${detected ? 'text-[#166534]' : 'text-[#374151]'}`}>
              {detected ? 'Planet Candidate Detected' : 'No Transit Signal'}
            </h3>
            <p className="text-sm text-[#6b7280]">TIC {ticId}</p>
          </div>
          {detected && (
            <div className="text-right">
              <p className="text-3xl font-bold text-[#166534]">{conf}%</p>
              <p className="text-xs text-[#6b7280]">confidence</p>
            </div>
          )}
        </div>

        {detected && (
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Orbital Period',   value: result.period_days    ? `${result.period_days.toFixed(3)} d`   : '—' },
              { label: 'Transit Depth',    value: result.depth_ppm      ? `${result.depth_ppm.toFixed(0)} ppm`  : '—' },
              { label: 'Transit Duration', value: result.duration_hours ? `${result.duration_hours.toFixed(2)} h` : '—' },
            ].map(({ label, value }) => (
              <div key={label} className="bg-white rounded-xl p-3 border border-[#d1fae5] text-center">
                <p className="text-xs text-[#6b7280] mb-1">{label}</p>
                <p className="text-base font-bold text-[#202124] font-mono">{value}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Vetting tests */}
      {result.vetting && (
        <div className="bg-white rounded-2xl border border-[#e5e7eb] p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm font-semibold text-[#374151]">Vetting Tests</p>
            <span className={`text-xs px-2.5 py-1 rounded-full font-semibold ${
              result.vetting.disposition === 'PLANET_CANDIDATE'
                ? 'bg-[#dcfce7] text-[#166534]'
                : result.vetting.disposition === 'LIKELY_FALSE_POSITIVE'
                ? 'bg-[#fee2e2] text-[#991b1b]'
                : 'bg-[#fef3c7] text-[#92400e]'
            }`}>
              {result.vetting.disposition.replace(/_/g, ' ')}
            </span>
          </div>
          <div className="space-y-2">
            <VettingRow label="Odd-Even Depth"       test={result.vetting.odd_even} />
            <VettingRow label="V-Shape Analysis"     test={result.vetting.v_shape} />
            <VettingRow label="Secondary Eclipse"    test={result.vetting.secondary_eclipse} />
          </div>
        </div>
      )}

      <button
        onClick={onReset}
        className="w-full flex items-center justify-center gap-2 py-3 bg-white hover:bg-[#f9fafb] text-[#374151] text-sm font-medium rounded-xl border border-[#e5e7eb] transition-colors"
      >
        <RotateCcw className="w-4 h-4" />
        Analyze Another Target
      </button>
    </div>
  )
}
