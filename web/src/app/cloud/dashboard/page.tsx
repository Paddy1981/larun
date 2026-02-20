'use client'

/**
 * Dashboard Page
 *
 * User dashboard with analysis history, usage statistics, and API key management.
 */

import { useState, useEffect, useCallback } from 'react'
import { getCurrentUser, getUserAnalyses, getUserQuota, supabase, type Analysis, type UsageQuota } from '@/lib/supabase'
import { QuotaIndicator } from '@/components/QuotaIndicator'
import { Loader2, Calendar, TrendingUp, Clock, Key, Plus, Trash2, Copy, Check, Eye, EyeOff } from 'lucide-react'
import Link from 'next/link'
import { format } from 'date-fns'

interface ApiKey {
  id: string
  key_prefix: string
  name: string
  plan: string
  calls_this_month: number
  calls_limit: number
  last_used_at: string | null
  created_at: string
  is_active: boolean
}

export default function DashboardPage() {
  const [user, setUser] = useState<any>(null)
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [quota, setQuota] = useState<UsageQuota | null>(null)
  const [loading, setLoading] = useState(true)

  // API key state
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([])
  const [keysLoading, setKeysLoading] = useState(false)
  const [newKeyName, setNewKeyName] = useState('')
  const [creatingKey, setCreatingKey] = useState(false)
  const [newKeySecret, setNewKeySecret] = useState<string | null>(null) // shown once
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [setupRequired, setSetupRequired] = useState(false)

  useEffect(() => {
    loadDashboard()
  }, [])

  const loadDashboard = async () => {
    const { user } = await getCurrentUser()
    if (!user) {
      window.location.href = '/auth/login?redirect=/dashboard'
      return
    }

    setUser(user)

    const [analysesData, quotaData] = await Promise.all([
      getUserAnalyses(user.id),
      getUserQuota(user.id),
    ])

    setAnalyses(analysesData)
    setQuota(quotaData)
    setLoading(false)

    // Load API keys after main data
    loadApiKeys()
  }

  const getAuthToken = async (): Promise<string | null> => {
    const { data: { session } } = await supabase.auth.getSession()
    return session?.access_token ?? null
  }

  const loadApiKeys = useCallback(async () => {
    setKeysLoading(true)
    try {
      const token = await getAuthToken()
      if (!token) return

      const res = await fetch('/api/v1/api-keys', {
        headers: { Authorization: `Bearer ${token}` },
      })
      const data = await res.json()
      if (data.setup_required) {
        setSetupRequired(true)
      } else {
        setApiKeys(data.keys ?? [])
      }
    } catch {
      // ignore
    } finally {
      setKeysLoading(false)
    }
  }, [])

  const createKey = async () => {
    if (creatingKey) return
    setCreatingKey(true)
    try {
      const token = await getAuthToken()
      if (!token) return

      const res = await fetch('/api/v1/api-keys', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: newKeyName.trim() || 'Default Key' }),
      })
      const data = await res.json()

      if (!res.ok) {
        if (data.setup_required) {
          setSetupRequired(true)
        } else {
          alert(data.error || 'Failed to create key')
        }
        return
      }

      setNewKeySecret(data.key) // show ONCE
      setNewKeyName('')
      await loadApiKeys()
    } finally {
      setCreatingKey(false)
    }
  }

  const revokeKey = async (id: string) => {
    if (!confirm('Revoke this API key? Any integrations using it will stop working.')) return
    const token = await getAuthToken()
    if (!token) return

    await fetch(`/api/v1/api-keys/${id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    })
    await loadApiKeys()
  }

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id)
      setTimeout(() => setCopiedId(null), 2000)
    })
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-larun-medium-gray" />
      </div>
    )
  }

  return (
    <div className="pt-24 pb-16 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl mb-2">Dashboard</h1>
          <p className="text-lg text-larun-medium-gray">
            Welcome back, {user?.email}
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {quota && (
            <QuotaIndicator quota={quota} />
          )}

          <div className="card">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-larun-lighter-gray flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-larun-medium-gray" />
              </div>
              <h4 className="text-lg font-medium">Total Analyses</h4>
            </div>
            <p className="text-3xl font-medium text-larun-black">
              {analyses.length}
            </p>
            <p className="text-sm text-larun-medium-gray mt-2">All time</p>
          </div>

          <div className="card">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-larun-lighter-gray flex items-center justify-center">
                <Clock className="w-5 h-5 text-larun-medium-gray" />
              </div>
              <h4 className="text-lg font-medium">Avg Inference</h4>
            </div>
            <p className="text-3xl font-medium text-larun-black">
              {analyses.length > 0
                ? (analyses.reduce((sum, a) => sum + a.inference_time_ms, 0) / analyses.length).toFixed(1)
                : '0'}{' '}
              <span className="text-lg text-larun-medium-gray">ms</span>
            </p>
            <p className="text-sm text-larun-medium-gray mt-2">Average speed</p>
          </div>
        </div>

        {/* ── API Keys ─────────────────────────────────────────────────────── */}
        <div className="card mb-10">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-larun-lighter-gray flex items-center justify-center">
                <Key className="w-5 h-5 text-larun-medium-gray" />
              </div>
              <div>
                <h2 className="text-2xl">API Keys</h2>
                <p className="text-sm text-larun-medium-gray">
                  Use these keys to call <code className="bg-larun-lighter-gray px-1 rounded text-xs">POST /api/tinyml/analyze</code> programmatically
                </p>
              </div>
            </div>
            <Link href="/cloud/pricing" className="text-sm text-larun-medium-gray underline">
              View plans
            </Link>
          </div>

          {/* One-time key reveal */}
          {newKeySecret && (
            <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
              <p className="text-sm font-semibold text-amber-800 mb-2">
                Save your key — it will not be shown again
              </p>
              <div className="flex items-center gap-2 bg-white border border-amber-300 rounded p-3 font-mono text-sm break-all">
                <span className="flex-1 select-all">{newKeySecret}</span>
                <button
                  onClick={() => copyToClipboard(newKeySecret, 'new')}
                  className="shrink-0 p-1 hover:text-amber-700"
                  title="Copy"
                >
                  {copiedId === 'new' ? <Check className="w-4 h-4 text-green-600" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>
              <button
                onClick={() => setNewKeySecret(null)}
                className="mt-2 text-xs text-amber-700 underline"
              >
                I've saved it — dismiss
              </button>
            </div>
          )}

          {/* Setup required notice */}
          {setupRequired && (
            <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm font-semibold text-blue-800 mb-1">One-time setup required</p>
              <p className="text-sm text-blue-700">
                Run the SQL migration in the{' '}
                <a href="https://supabase.com/dashboard" target="_blank" rel="noopener noreferrer" className="underline">
                  Supabase SQL Editor
                </a>{' '}
                to enable API keys. File: <code className="bg-blue-100 px-1 rounded">supabase/migrations/001_api_keys.sql</code>
              </p>
            </div>
          )}

          {/* Create new key */}
          <div className="flex gap-3 mb-6">
            <input
              type="text"
              placeholder="Key name (e.g. My Pipeline)"
              value={newKeyName}
              onChange={e => setNewKeyName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && createKey()}
              className="flex-1 border border-larun-light-gray rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-larun-black"
              maxLength={80}
            />
            <button
              onClick={createKey}
              disabled={creatingKey}
              className="btn btn-primary flex items-center gap-2 disabled:opacity-50"
            >
              {creatingKey ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
              Generate Key
            </button>
          </div>

          {/* Key list */}
          {keysLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-larun-medium-gray" />
            </div>
          ) : apiKeys.length === 0 && !setupRequired ? (
            <p className="text-sm text-larun-medium-gray text-center py-8">
              No API keys yet. Generate one above to get programmatic access.
            </p>
          ) : (
            <div className="space-y-3">
              {apiKeys.filter(k => k.is_active).map(key => (
                <div
                  key={key.id}
                  className="flex items-center justify-between p-4 border border-larun-light-gray rounded-lg"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-sm truncate">{key.name}</span>
                      <span className="text-xs bg-larun-lighter-gray px-2 py-0.5 rounded-full capitalize">
                        {key.plan}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-larun-medium-gray">
                      <span className="font-mono">{key.key_prefix}••••••••</span>
                      <span>
                        {key.calls_this_month.toLocaleString()}&nbsp;/&nbsp;
                        {key.calls_limit === -1 ? '∞' : key.calls_limit.toLocaleString()} calls this month
                      </span>
                      {key.last_used_at && (
                        <span>Last used {format(new Date(key.last_used_at), 'MMM d')}</span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => revokeKey(key.id)}
                    className="ml-4 p-2 text-larun-medium-gray hover:text-red-500 transition-colors"
                    title="Revoke key"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Usage example */}
          <div className="mt-6 p-4 bg-larun-lighter-gray rounded-lg">
            <p className="text-xs font-semibold text-larun-medium-gray mb-2">QUICK START</p>
            <pre className="text-xs text-larun-dark-gray overflow-x-auto whitespace-pre-wrap">{`curl -X POST https://larun.space/api/tinyml/analyze \\
  -H "X-API-Key: lrn_live_<your-key>" \\
  -F "file=@lightcurve.fits" \\
  -F "model_id=EXOPLANET-001"`}</pre>
          </div>
        </div>

        {/* ── Analysis History ─────────────────────────────────────────────── */}
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl">Recent Analyses</h2>
            <Link href="/cloud/analyze" className="btn btn-primary">
              New Analysis
            </Link>
          </div>

          {analyses.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 rounded-full bg-larun-lighter-gray flex items-center justify-center mx-auto mb-4">
                <Calendar className="w-8 h-8 text-larun-medium-gray" />
              </div>
              <p className="text-larun-medium-gray mb-6">
                No analyses yet. Upload a FITS file to get started.
              </p>
              <Link href="/cloud/analyze" className="btn btn-primary">
                Start Analyzing
              </Link>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-larun-light-gray text-left">
                    <th className="pb-3 text-sm font-medium text-larun-medium-gray">Date</th>
                    <th className="pb-3 text-sm font-medium text-larun-medium-gray">Model</th>
                    <th className="pb-3 text-sm font-medium text-larun-medium-gray">Classification</th>
                    <th className="pb-3 text-sm font-medium text-larun-medium-gray">Confidence</th>
                    <th className="pb-3 text-sm font-medium text-larun-medium-gray">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {analyses.slice(0, 20).map((analysis) => (
                    <tr key={analysis.id} className="border-b border-larun-lighter-gray">
                      <td className="py-4 text-sm text-larun-dark-gray">
                        {format(new Date(analysis.created_at), 'MMM d, yyyy HH:mm')}
                      </td>
                      <td className="py-4 text-sm font-medium text-larun-black">
                        {analysis.model_id}
                      </td>
                      <td className="py-4">
                        <span className="inline-flex px-3 py-1 rounded-full text-xs font-medium bg-larun-lighter-gray text-larun-black">
                          {analysis.classification.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="py-4 text-sm text-larun-dark-gray">
                        {(analysis.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-4 text-sm text-larun-dark-gray">
                        {analysis.inference_time_ms.toFixed(1)} ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
