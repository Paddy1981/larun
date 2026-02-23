'use client'

import { useState, useEffect } from 'react'
import { FileUpload } from '@/components/FileUpload'
import { ModelSelector } from '@/components/ModelSelector'
import { ResultsDisplay } from '@/components/ResultsDisplay'
import { QuotaIndicator } from '@/components/QuotaIndicator'
import { ColorIndexInput, type ColorIndices } from '@/components/ColorIndexInput'
import { getCurrentUser, getUserQuota, type UsageQuota } from '@/lib/supabase'
import { apiClient, getModelById } from '@/lib/api-client'
import type { InferenceResult } from '@/lib/supabase'
import { Loader2, AlertCircle, Search, Upload } from 'lucide-react'
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
  { id: '470710327', name: 'TOI-1338 b', description: 'Circumbinary planet' },
  { id: '307210830', name: 'TOI-700 d',  description: 'Earth-sized in HZ' },
  { id: '261136679', name: 'TOI-175 b',  description: 'Super-Earth' },
]

const EMPTY_INDICES: ColorIndices = { bv: '', vr: '', bp_rp: '', jh: '', hk: '' }

// ── Component ─────────────────────────────────────────────────────────────────

export default function AnalyzePage() {
  const [user, setUser] = useState<any>(null)
  const [quota, setQuota] = useState<UsageQuota | null>(null)

  // Mode: 'tic' = TIC ID lookup, 'upload' = FITS/colour index
  const [mode, setMode] = useState<'tic' | 'upload'>('tic')

  // TIC ID mode state
  const [ticId, setTicId] = useState('')
  const [ticLoading, setTicLoading] = useState(false)
  const [ticProgress, setTicProgress] = useState(0)
  const [ticResult, setTicResult] = useState<TicResult | null>(null)
  const [ticError, setTicError] = useState<string | null>(null)
  const [ticId_used, setTicIdUsed] = useState('')   // for showing in result header

  // Upload mode state
  const [selectedModel, setSelectedModel] = useState('EXOPLANET-001')
  const [file, setFile] = useState<File | null>(null)
  const [colorIndices, setColorIndices] = useState<ColorIndices>(EMPTY_INDICES)
  const [uploadResult, setUploadResult] = useState<InferenceResult | null>(null)
  const [uploadLoading, setUploadLoading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  useEffect(() => { checkAuth() }, [])

  const checkAuth = async () => {
    const { user } = await getCurrentUser()
    if (!user) {
      window.location.href = '/cloud/auth/login?redirect=/cloud/analyze'
      return
    }
    setUser(user)
    setQuota(await getUserQuota(user.id))
  }

  const isQuotaExceeded =
    quota !== null && quota.quota_limit !== null &&
    quota.quota_limit !== -1 && quota.analyses_count >= quota.quota_limit

  const refreshQuota = async () => {
    if (user) setQuota(await getUserQuota(user.id))
  }

  // ── TIC ID analysis ─────────────────────────────────────────────────────────
  const handleTicAnalyze = async (targetId?: string) => {
    const id = (targetId ?? ticId).trim()
    if (!id || !user) return

    setTicLoading(true)
    setTicError(null)
    setTicResult(null)
    setTicProgress(0)
    setTicIdUsed(id)

    const tick = setInterval(() => setTicProgress(p => Math.min(85, p + 5)), 1500)

    try {
      const res = await fetch('/api/v1/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tic_id: id, user_id: user.id }),
      })

      clearInterval(tick)
      setTicProgress(100)

      const text = await res.text()
      let data: any
      try { data = JSON.parse(text) } catch {
        throw new Error(res.status === 504
          ? 'Analysis timed out fetching TESS data. Try again or pick a different target.'
          : `Server error (${res.status}). Please try again.`)
      }

      if (!res.ok) throw new Error(data.error?.message || 'Analysis failed')
      if (data.status === 'failed') throw new Error(data.error || 'Detection failed')

      setTicResult(data.result)
      await refreshQuota()
    } catch (err: any) {
      setTicError(err.message || 'An error occurred')
    } finally {
      clearInterval(tick)
      setTicLoading(false)
    }
  }

  // ── Upload / colour index analysis ─────────────────────────────────────────
  const isSpectral = selectedModel === 'SPECTYPE-001'
  const hasColorIndex = Object.values(colorIndices).some(v => v.trim() !== '' && !isNaN(Number(v)))
  const canRunUpload = isSpectral ? hasColorIndex : !!file

  const handleUploadAnalyze = async () => {
    if (!user || !canRunUpload || isQuotaExceeded) return

    setUploadLoading(true)
    setUploadError(null)
    setUploadResult(null)

    try {
      let res: InferenceResult
      if (isSpectral) {
        const parse = (v: string) => v.trim() === '' ? undefined : Number(v)
        res = await apiClient.analyzeSpectralType(
          { bv: parse(colorIndices.bv), vr: parse(colorIndices.vr),
            bp_rp: parse(colorIndices.bp_rp), jh: parse(colorIndices.jh), hk: parse(colorIndices.hk) },
          user.id
        )
      } else {
        res = await apiClient.analyzeTinyML(file!, selectedModel, user.id)
      }
      setUploadResult(res)
      await refreshQuota()
    } catch (err: any) {
      setUploadError(err.response?.data?.detail || err.message || 'Analysis failed. Please try again.')
    } finally {
      setUploadLoading(false)
    }
  }

  // ── Loading guard ───────────────────────────────────────────────────────────
  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-larun-medium-gray" />
      </div>
    )
  }

  const selectedModelMeta = getModelById(selectedModel)

  return (
    <div className="pt-24 pb-16 px-6">
      <div className="max-w-6xl mx-auto">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl mb-3">Analyze</h1>
          <p className="text-lg text-larun-medium-gray">
            Look up a TESS target by ID, or upload your own data file.
          </p>
        </div>

        {/* Quota */}
        {quota && <QuotaIndicator quota={quota} className="mb-8" />}

        {/* Quota exceeded */}
        {isQuotaExceeded && (
          <div className="mb-8 bg-amber-50 border border-amber-200 rounded-lg p-5 flex items-start gap-4">
            <AlertCircle className="w-6 h-6 text-amber-500 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-amber-900 mb-1">Monthly quota reached</p>
              <p className="text-sm text-amber-800 mb-3">
                You have used all {quota?.quota_limit} analyses this month.
              </p>
              <Link href="/cloud/pricing" className="btn btn-primary btn-sm">View Plans →</Link>
            </div>
          </div>
        )}

        {/* Mode toggle */}
        <div className="flex gap-2 mb-8 p-1 bg-larun-lighter-gray rounded-xl w-fit">
          <button
            onClick={() => { setMode('tic'); setTicResult(null); setTicError(null) }}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
              mode === 'tic'
                ? 'bg-white text-larun-black shadow-sm'
                : 'text-larun-medium-gray hover:text-larun-black'
            }`}
          >
            <Search className="w-4 h-4" />
            TIC ID Lookup
          </button>
          <button
            onClick={() => { setMode('upload'); setUploadResult(null); setUploadError(null) }}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
              mode === 'upload'
                ? 'bg-white text-larun-black shadow-sm'
                : 'text-larun-medium-gray hover:text-larun-black'
            }`}
          >
            <Upload className="w-4 h-4" />
            Upload File
          </button>
        </div>

        {/* ── TIC ID MODE ────────────────────────────────────────────────────── */}
        {mode === 'tic' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="card">
                <h3 className="text-xl mb-1">TESS Target Lookup</h3>
                <p className="text-sm text-larun-medium-gray mb-5">
                  Enter a TESS Input Catalog ID. We'll fetch the light curve from NASA MAST and run BLS transit detection automatically.
                </p>

                {/* Input */}
                <div className="flex gap-3 mb-4">
                  <input
                    type="text"
                    placeholder="TIC ID (e.g. 470710327)"
                    value={ticId}
                    onChange={e => setTicId(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && !ticLoading && ticId.trim() && handleTicAnalyze()}
                    disabled={ticLoading || isQuotaExceeded}
                    className="input flex-1 font-mono"
                  />
                  <button
                    onClick={() => handleTicAnalyze()}
                    disabled={!ticId.trim() || ticLoading || isQuotaExceeded}
                    className="btn btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {ticLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Analyze'}
                  </button>
                </div>

                {/* Popular targets */}
                {!ticLoading && !ticResult && (
                  <div>
                    <p className="text-xs text-larun-medium-gray mb-2">Try a known target:</p>
                    <div className="flex flex-wrap gap-2">
                      {POPULAR_TARGETS.map(t => (
                        <button
                          key={t.id}
                          onClick={() => { setTicId(t.id); handleTicAnalyze(t.id) }}
                          disabled={isQuotaExceeded}
                          className="text-xs px-3 py-1.5 bg-larun-lighter-gray hover:bg-larun-light-gray rounded-full transition-colors disabled:opacity-50"
                        >
                          {t.name} <span className="text-larun-medium-gray">({t.description})</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {ticError && (
                  <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                    <AlertCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                    <p className="text-sm text-red-700">{ticError}</p>
                  </div>
                )}
              </div>

              {/* How it works */}
              {!ticLoading && !ticResult && (
                <div className="card">
                  <h4 className="font-medium mb-4">How it works</h4>
                  <ol className="space-y-3">
                    {[
                      ['Fetch', 'Light curve data retrieved from NASA TESS archives via MAST'],
                      ['Detect', 'BLS periodogram searches for periodic transit-shaped dips'],
                      ['Vet', 'Odd-even, V-shape, and secondary eclipse tests flag false positives'],
                    ].map(([title, desc]) => (
                      <li key={title} className="flex gap-3 text-sm">
                        <span className="w-6 h-6 rounded-full bg-larun-black text-white flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">
                          {['Fetch','Detect','Vet'].indexOf(title) + 1}
                        </span>
                        <div>
                          <span className="font-medium text-larun-black">{title}</span>
                          <span className="text-larun-medium-gray"> — {desc}</span>
                        </div>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </div>

            {/* TIC Results */}
            <div>
              {ticLoading ? (
                <div className="card flex flex-col items-center justify-center min-h-[340px] text-center">
                  <div className="w-16 h-16 relative mb-4">
                    <div className="absolute inset-0 border-4 border-larun-lighter-gray rounded-full" />
                    <div className="absolute inset-0 border-4 border-larun-black rounded-full border-t-transparent animate-spin" />
                  </div>
                  <p className="font-medium mb-1">Analyzing TIC {ticId_used}…</p>
                  <p className="text-sm text-larun-medium-gray mb-4">Fetching TESS data and running transit detection</p>
                  <div className="w-full max-w-xs bg-larun-lighter-gray rounded-full h-1.5">
                    <div className="bg-larun-black h-1.5 rounded-full transition-all duration-500" style={{ width: `${ticProgress}%` }} />
                  </div>
                  <p className="text-xs text-larun-medium-gray mt-2">{ticProgress}%</p>
                </div>
              ) : ticResult ? (
                <TicResultCard result={ticResult} ticId={ticId_used} onReset={() => { setTicResult(null); setTicId('') }} />
              ) : (
                <div className="card h-full flex items-center justify-center min-h-[340px]">
                  <div className="text-center text-larun-medium-gray">
                    <Search className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <p className="font-medium">Results will appear here</p>
                    <p className="text-sm mt-1">Enter a TIC ID and run analysis</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── UPLOAD MODE ────────────────────────────────────────────────────── */}
        {mode === 'upload' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="card">
                <h3 className="text-xl mb-1">1. Choose a Model</h3>
                <p className="text-sm text-larun-medium-gray mb-5">
                  Pick the model that matches what you are looking for in your data.
                </p>
                <ModelSelector
                  selectedModel={selectedModel}
                  onModelSelect={id => { setSelectedModel(id); setUploadResult(null); setUploadError(null) }}
                  disabled={uploadLoading}
                />
              </div>

              <div className="card">
                {isSpectral ? (
                  <>
                    <h3 className="text-xl mb-1">2. Enter Colour Indices</h3>
                    <ColorIndexInput
                      values={colorIndices}
                      onChange={setColorIndices}
                      disabled={uploadLoading || isQuotaExceeded}
                    />
                  </>
                ) : (
                  <>
                    <h3 className="text-xl mb-1">2. Upload Data File</h3>
                    {selectedModelMeta && (
                      <p className="text-sm text-larun-medium-gray mb-4">
                        <span className="font-medium text-larun-black">Expected source:</span>{' '}
                        {selectedModelMeta.data_source}
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

              <button
                onClick={handleUploadAnalyze}
                disabled={!canRunUpload || uploadLoading || isQuotaExceeded}
                className="btn btn-primary w-full text-lg py-4 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploadLoading ? (
                  <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing…</>
                ) : '3. Run Analysis'}
              </button>

              {uploadError && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-red-800 mb-1">Analysis Failed</p>
                    <p className="text-sm text-red-700">{uploadError}</p>
                    {uploadError.includes('quota') && (
                      <Link href="/cloud/pricing" className="text-sm text-red-800 underline mt-2 inline-block">
                        View Plans →
                      </Link>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div>
              {uploadResult ? (
                <ResultsDisplay result={uploadResult} modelId={selectedModel} />
              ) : (
                <div className="card h-full flex items-center justify-center min-h-[400px]">
                  <div className="text-center text-larun-medium-gray">
                    <Upload className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <p className="font-medium">Results will appear here</p>
                    <p className="text-sm mt-1">
                      {isSpectral ? 'Enter colour indices above' : 'Upload a file and run analysis'}
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

// ── TIC Result Card ────────────────────────────────────────────────────────────

function VettingRow({ label, test }: { label: string; test: VettingTest }) {
  const colors = {
    PASS:    'bg-green-100 text-green-700',
    FAIL:    'bg-red-100 text-red-700',
    WARNING: 'bg-amber-100 text-amber-700',
  }
  return (
    <div className="flex items-center justify-between p-3 bg-larun-lighter-gray rounded-lg">
      <div>
        <p className="text-sm font-medium text-larun-black">{label}</p>
        <p className="text-xs text-larun-medium-gray">{test.message}</p>
      </div>
      <span className={`text-xs px-2 py-0.5 rounded-full font-medium shrink-0 ml-3 ${colors[test.flag]}`}>
        {test.flag}
      </span>
    </div>
  )
}

function TicResultCard({
  result, ticId, onReset
}: { result: TicResult; ticId: string; onReset: () => void }) {
  return (
    <div className="space-y-4">
      {/* Detection summary */}
      <div className={`card ${result.detection ? 'border-green-300 bg-green-50' : ''}`}>
        <div className="flex items-center gap-4 mb-4">
          <div className={`w-12 h-12 rounded-full flex items-center justify-center shrink-0 ${
            result.detection ? 'bg-green-100' : 'bg-larun-lighter-gray'
          }`}>
            {result.detection ? (
              <svg className="w-7 h-7 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-7 h-7 text-larun-medium-gray" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            )}
          </div>
          <div>
            <h3 className={`text-xl font-semibold ${result.detection ? 'text-green-700' : 'text-larun-black'}`}>
              {result.detection ? 'Planet Candidate Detected' : 'No Transit Signal Detected'}
            </h3>
            <p className="text-sm text-larun-medium-gray">TIC {ticId}</p>
          </div>
        </div>

        {result.detection && (
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Confidence', value: `${(result.confidence * 100).toFixed(1)}%` },
              { label: 'Period', value: result.period_days ? `${result.period_days.toFixed(2)} days` : '—' },
              { label: 'Depth', value: result.depth_ppm ? `${result.depth_ppm.toFixed(0)} ppm` : '—' },
              { label: 'Duration', value: result.duration_hours ? `${result.duration_hours.toFixed(1)} hrs` : '—' },
            ].map(({ label, value }) => (
              <div key={label} className="bg-white rounded-lg p-3 border border-green-200">
                <p className="text-xs text-larun-medium-gray uppercase tracking-wide mb-1">{label}</p>
                <p className="text-lg font-bold text-larun-black">{value}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Vetting */}
      {result.vetting && (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold">Vetting Tests</h4>
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              result.vetting.disposition === 'PLANET_CANDIDATE'    ? 'bg-green-100 text-green-700' :
              result.vetting.disposition === 'LIKELY_FALSE_POSITIVE' ? 'bg-red-100 text-red-700' :
              'bg-amber-100 text-amber-700'
            }`}>
              {result.vetting.disposition.replace(/_/g, ' ')}
            </span>
          </div>
          <div className="space-y-2">
            <VettingRow label="Odd-Even Depth Test"    test={result.vetting.odd_even} />
            <VettingRow label="V-Shape Analysis"       test={result.vetting.v_shape} />
            <VettingRow label="Secondary Eclipse Check" test={result.vetting.secondary_eclipse} />
          </div>
        </div>
      )}

      <button
        onClick={onReset}
        className="btn btn-outline w-full"
      >
        Analyze Another Target
      </button>
    </div>
  )
}
