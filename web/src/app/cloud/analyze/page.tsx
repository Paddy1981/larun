'use client'

/**
 * Analysis Page
 *
 * Main interface for uploading FITS files and running TinyML inference
 */

import { useState, useEffect } from 'react'
import { FileUpload } from '@/components/FileUpload'
import { ModelSelector } from '@/components/ModelSelector'
import { ResultsDisplay } from '@/components/ResultsDisplay'
import { QuotaIndicator } from '@/components/QuotaIndicator'
import { getCurrentUser, getUserQuota, getUserSubscription, type UsageQuota, type Subscription } from '@/lib/supabase'
import { apiClient } from '@/lib/api-client'
import type { InferenceResult } from '@/lib/supabase'
import { Loader2, AlertCircle } from 'lucide-react'
import Link from 'next/link'

export default function AnalyzePage() {
  const [user, setUser] = useState<any>(null)
  const [quota, setQuota] = useState<UsageQuota | null>(null)
  const [subscription, setSubscription] = useState<Subscription | null | undefined>(undefined)
  const [selectedModel, setSelectedModel] = useState('EXOPLANET-001')
  const [file, setFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [error, setError] = useState<string | null>(null)

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

    // Load quota and subscription in parallel
    const [quotaData, subData] = await Promise.all([
      getUserQuota(user.id),
      getUserSubscription(user.id),
    ])
    setQuota(quotaData)
    setSubscription(subData)
  }

  const handleAnalyze = async () => {
    if (!file || !user) return

    // Guard: active subscription required
    if (!subscription) {
      setError('An active subscription is required to run cloud models. Please subscribe to continue.')
      return
    }

    // Check quota (skip if unlimited: analyses_limit === -1)
    if (quota && quota.quota_limit !== null && quota.quota_limit !== -1) {
      if (quota.analyses_count >= quota.quota_limit) {
        setError('You have reached your monthly analysis quota. Please upgrade your plan.')
        return
      }
    }

    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    try {
      const result = await apiClient.analyzeTinyML(file, selectedModel, user.id)
      setResult(result)

      // Refresh quota
      const updatedQuota = await getUserQuota(user.id)
      setQuota(updatedQuota)
    } catch (err: any) {
      console.error('Analysis failed:', err)
      const errorMessage = err.response?.data?.detail || err.message || 'Analysis failed. Please try again.'
      setError(errorMessage)
    } finally {
      setIsAnalyzing(false)
    }
  }

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
        <div className="mb-12">
          <h1 className="text-4xl mb-4">Analyze Light Curves</h1>
          <p className="text-lg text-larun-medium-gray">
            Upload FITS files and run TinyML inference with 8 specialized models
          </p>
        </div>

        {/* Subscription gate banner */}
        {subscription === null && (
          <div className="mb-8 bg-amber-50 border border-amber-200 rounded-lg p-5 flex items-start gap-4">
            <AlertCircle className="w-6 h-6 text-amber-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-amber-900 mb-1">Subscription required</p>
              <p className="text-sm text-amber-800 mb-3">
                Running cloud models requires an active subscription. Subscribe to unlock full access.
              </p>
              <Link href="/cloud/pricing" className="btn btn-primary btn-sm">
                View Plans →
              </Link>
            </div>
          </div>
        )}

        {/* Quota Indicator */}
        {quota && <QuotaIndicator quota={quota} className="mb-8" />}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Upload & Settings */}
          <div className="space-y-6">
            {/* File Upload */}
            <div className="card">
              <h3 className="text-xl mb-4">1. Upload FITS File</h3>
              <FileUpload
                onFileSelect={setFile}
                selectedFile={file}
                disabled={isAnalyzing || !subscription}
              />
            </div>

            {/* Model Selection */}
            <div className="card">
              <h3 className="text-xl mb-4">2. Select Model</h3>
              <ModelSelector
                selectedModel={selectedModel}
                onModelSelect={setSelectedModel}
                disabled={isAnalyzing || !subscription}
              />
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={!file || isAnalyzing || !subscription}
              className="btn btn-primary w-full text-lg py-4 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : !subscription ? (
                'Subscribe to Run Analysis'
              ) : (
                '3. Run Analysis'
              )}
            </button>

            {/* Error Display */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-800 mb-1">
                    Analysis Failed
                  </p>
                  <p className="text-sm text-red-700">{error}</p>
                  {(error.includes('quota') || error.includes('subscription')) && (
                    <Link
                      href="/cloud/pricing"
                      className="text-sm text-red-800 underline mt-2 inline-block"
                    >
                      View Plans →
                    </Link>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          <div>
            {result ? (
              <ResultsDisplay result={result} modelId={selectedModel} />
            ) : (
              <div className="card h-full flex items-center justify-center min-h-[400px]">
                <div className="text-center text-larun-medium-gray">
                  <div className="w-24 h-24 rounded-full bg-larun-lighter-gray flex items-center justify-center mx-auto mb-4">
                    <svg
                      className="w-12 h-12 text-larun-medium-gray"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                      />
                    </svg>
                  </div>
                  <p className="font-medium">Results will appear here</p>
                  <p className="text-sm mt-2">
                    Upload a FITS file and run analysis to see results
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
