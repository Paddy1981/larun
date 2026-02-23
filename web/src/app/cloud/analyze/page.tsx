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
import { Loader2, AlertCircle } from 'lucide-react'
import Link from 'next/link'

const EMPTY_INDICES: ColorIndices = { bv: '', vr: '', bp_rp: '', jh: '', hk: '' }

export default function AnalyzePage() {
  const [user, setUser] = useState<any>(null)
  const [quota, setQuota] = useState<UsageQuota | null>(null)
  const [selectedModel, setSelectedModel] = useState('EXOPLANET-001')

  // FITS flow
  const [file, setFile] = useState<File | null>(null)

  // SPECTYPE-001 colour index flow
  const [colorIndices, setColorIndices] = useState<ColorIndices>(EMPTY_INDICES)

  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const isSpectral = selectedModel === 'SPECTYPE-001'

  useEffect(() => {
    checkAuth()
  }, [])

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
    quota !== null &&
    quota.quota_limit !== null &&
    quota.quota_limit !== -1 &&
    quota.analyses_count >= quota.quota_limit

  const hasColorIndex = Object.values(colorIndices).some(
    v => v.trim() !== '' && !isNaN(Number(v))
  )

  // True when the user has provided enough input to run
  const canRun = isSpectral ? hasColorIndex : !!file

  const handleModelChange = (id: string) => {
    setSelectedModel(id)
    setResult(null)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!user || isQuotaExceeded) return

    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    try {
      let res: InferenceResult

      if (isSpectral) {
        // Parse colour index strings to numbers (undefined if blank)
        const parse = (v: string) => (v.trim() === '' ? undefined : Number(v))
        res = await apiClient.analyzeSpectralType(
          {
            bv:    parse(colorIndices.bv),
            vr:    parse(colorIndices.vr),
            bp_rp: parse(colorIndices.bp_rp),
            jh:    parse(colorIndices.jh),
            hk:    parse(colorIndices.hk),
          },
          user.id
        )
      } else {
        if (!file) return
        res = await apiClient.analyzeTinyML(file, selectedModel, user.id)
      }

      setResult(res)
      setQuota(await getUserQuota(user.id))
    } catch (err: any) {
      console.error('Analysis failed:', err)
      setError(err.response?.data?.detail || err.message || 'Analysis failed. Please try again.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const selectedModelMeta = getModelById(selectedModel)

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-larun-medium-gray" />
      </div>
    )
  }

  return (
    <div className="pt-24 pb-16 px-6">
      <div className="max-w-6xl mx-auto">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl mb-3">Analyze</h1>
          <p className="text-lg text-larun-medium-gray">
            {isSpectral
              ? 'Enter photometric colour indices to classify stellar spectral type'
              : 'Upload a FITS or ASCII time-series file and run one of 8 specialist TinyML models'}
          </p>
        </div>

        {/* Quota indicator */}
        {quota && <QuotaIndicator quota={quota} className="mb-8" />}

        {/* Quota exceeded banner */}
        {isQuotaExceeded && (
          <div className="mb-8 bg-amber-50 border border-amber-200 rounded-lg p-5 flex items-start gap-4">
            <AlertCircle className="w-6 h-6 text-amber-500 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-amber-900 mb-1">Monthly quota reached</p>
              <p className="text-sm text-amber-800 mb-3">
                You have used all {quota?.quota_limit} analyses for this month. Upgrade to continue.
              </p>
              <Link href="/cloud/pricing" className="btn btn-primary btn-sm">View Plans →</Link>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left column */}
          <div className="space-y-6">

            {/* Step 1: Model selection */}
            <div className="card">
              <h3 className="text-xl mb-1">1. Choose a Model</h3>
              <p className="text-sm text-larun-medium-gray mb-5">
                Pick the model that matches what you are looking for in your data.
              </p>
              <ModelSelector
                selectedModel={selectedModel}
                onModelSelect={handleModelChange}
                disabled={isAnalyzing}
              />
            </div>

            {/* Step 2: Input — varies by model */}
            <div className="card">
              {isSpectral ? (
                <>
                  <h3 className="text-xl mb-1">2. Enter Colour Indices</h3>
                  <ColorIndexInput
                    values={colorIndices}
                    onChange={setColorIndices}
                    disabled={isAnalyzing || isQuotaExceeded}
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
                    disabled={isAnalyzing || isQuotaExceeded}
                  />
                </>
              )}
            </div>

            {/* Step 3: Run */}
            <button
              onClick={handleAnalyze}
              disabled={!canRun || isAnalyzing || isQuotaExceeded}
              className="btn btn-primary w-full text-lg py-4 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing…
                </>
              ) : (
                '3. Run Analysis'
              )}
            </button>

            {/* Error */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-800 mb-1">Analysis Failed</p>
                  <p className="text-sm text-red-700">{error}</p>
                  {error.includes('quota') && (
                    <Link href="/cloud/pricing" className="text-sm text-red-800 underline mt-2 inline-block">
                      View Plans →
                    </Link>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right column: Results */}
          <div>
            {result ? (
              <ResultsDisplay result={result} modelId={selectedModel} />
            ) : (
              <div className="card h-full flex items-center justify-center min-h-[400px]">
                <div className="text-center text-larun-medium-gray">
                  <div className="w-24 h-24 rounded-full bg-larun-lighter-gray flex items-center justify-center mx-auto mb-4">
                    <svg className="w-12 h-12 text-larun-medium-gray" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                      />
                    </svg>
                  </div>
                  <p className="font-medium">Results will appear here</p>
                  <p className="text-sm mt-2">
                    {isSpectral
                      ? 'Enter colour indices above and run analysis'
                      : 'Select a model, upload a file, and run analysis'}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
