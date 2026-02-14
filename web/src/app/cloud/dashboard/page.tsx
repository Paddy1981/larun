'use client'

/**
 * Dashboard Page
 *
 * User dashboard with analysis history and usage statistics
 */

import { useState, useEffect } from 'react'
import { getCurrentUser, getUserAnalyses, getUserQuota, type Analysis, type UsageQuota } from '@/lib/supabase'
import { QuotaIndicator } from '@/components/QuotaIndicator'
import { Loader2, Calendar, TrendingUp, Clock } from 'lucide-react'
import Link from 'next/link'
import { format } from 'date-fns'

export default function DashboardPage() {
  const [user, setUser] = useState<any>(null)
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [quota, setQuota] = useState<UsageQuota | null>(null)
  const [loading, setLoading] = useState(true)

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
            <p className="text-sm text-larun-medium-gray mt-2">
              All time
            </p>
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
            <p className="text-sm text-larun-medium-gray mt-2">
              Average speed
            </p>
          </div>
        </div>

        {/* Analysis History */}
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl">Recent Analyses</h2>
            <Link href="/analyze" className="btn btn-primary">
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
              <Link href="/analyze" className="btn btn-primary">
                Start Analyzing
              </Link>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-larun-light-gray text-left">
                    <th className="pb-3 text-sm font-medium text-larun-medium-gray">
                      Date
                    </th>
                    <th className="pb-3 text-sm font-medium text-larun-medium-gray">
                      Model
                    </th>
                    <th className="pb-3 text-sm font-medium text-larun-medium-gray">
                      Classification
                    </th>
                    <th className="pb-3 text-sm font-medium text-larun-medium-gray">
                      Confidence
                    </th>
                    <th className="pb-3 text-sm font-medium text-larun-medium-gray">
                      Time
                    </th>
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
