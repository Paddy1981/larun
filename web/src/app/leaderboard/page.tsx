'use client'

/**
 * /leaderboard — Discovery leaderboard page
 */

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { useSession } from 'next-auth/react'
import Header from '@/components/Header'
import { LeaderboardTable, RankLegend } from '@/components/discovery/LeaderboardTable'
import { discoveryClient, Leaderboard, UserStats } from '@/lib/discovery-client'
import { Trophy, User, Telescope, Loader2, RefreshCw } from 'lucide-react'

function StatCard({ label, value, icon: Icon, color }: {
  label: string
  value: string | number
  icon: typeof Trophy
  color: string
}) {
  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm px-6 py-5 flex items-center gap-4">
      <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
        style={{ backgroundColor: `${color}15` }}>
        <Icon className="w-5 h-5" style={{ color }} />
      </div>
      <div>
        <p className="text-xl font-semibold text-[#202124]">{value}</p>
        <p className="text-xs text-[#9ca3af]">{label}</p>
      </div>
    </div>
  )
}

function UserStatsCard({ stats }: { stats: UserStats }) {
  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-5">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-9 h-9 rounded-full bg-[#e8f0fe] flex items-center justify-center">
          <User className="w-4 h-4 text-[#1a73e8]" />
        </div>
        <div>
          <p className="text-sm font-medium text-[#202124]">{stats.user_id}</p>
          <p className="text-xs text-[#9ca3af]">{stats.rank}</p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div className="text-center bg-[#f8f9fa] rounded-xl py-3">
          <p className="text-lg font-semibold text-[#202124]">{stats.verified_discoveries}</p>
          <p className="text-xs text-[#9ca3af]">verified</p>
        </div>
        <div className="text-center bg-[#f8f9fa] rounded-xl py-3">
          <p className="text-lg font-semibold text-[#1a73e8]">{stats.total_submissions}</p>
          <p className="text-xs text-[#9ca3af]">submitted</p>
        </div>
        <div className="text-center bg-[#f8f9fa] rounded-xl py-3">
          <p className="text-lg font-semibold text-[#f59e0b]">{(stats.points ?? 0).toLocaleString()}</p>
          <p className="text-xs text-[#9ca3af]">points</p>
        </div>
      </div>
    </div>
  )
}

// Empty state used when backend is unreachable
const EMPTY_DATA: Leaderboard = {
  rankings: [],
  total_users: 0,
  total_discoveries: 0,
  period: 'All time',
}

export default function LeaderboardPage() {
  const { data: session } = useSession()
  const [leaderboard, setLeaderboard] = useState<Leaderboard>(EMPTY_DATA)
  const [loading, setLoading] = useState(true)
  const [backendOffline, setBackendOffline] = useState(false)

  const currentUserId = session?.user?.email ?? session?.user?.name ?? undefined

  useEffect(() => {
    loadLeaderboard()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function loadLeaderboard() {
    setLoading(true)
    setBackendOffline(false)
    try {
      const data = await discoveryClient.getLeaderboard()
      setLeaderboard(data)
    } catch {
      setLeaderboard(EMPTY_DATA)
      setBackendOffline(true)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#f8f9fa]">
      <Header />

      <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-24 pb-16">
        {/* Page header */}
        <div className="mb-8 flex items-start justify-between gap-4 flex-wrap">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Trophy className="w-5 h-5 text-[#f59e0b]" />
              <h1 className="text-2xl font-semibold text-[#202124]">Discovery Leaderboard</h1>
            </div>
            <p className="text-sm text-[#5f6368]">
              Top citizen astronomers ranked by verified space object discoveries.
            </p>
            {backendOffline && (
              <p className="text-xs text-[#d97706] mt-1">
                Backend offline — rankings will appear here once the discovery API is connected.
              </p>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={loadLeaderboard}
              disabled={loading}
              className="btn btn-outline py-2 px-4 flex items-center gap-2 text-sm"
            >
              {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
              Refresh
            </button>
            <Link href="/discover" className="btn btn-primary py-2 px-4 flex items-center gap-2 text-sm">
              <Telescope className="w-3.5 h-3.5" />
              Start Discovering
            </Link>
          </div>
        </div>

        {/* Global stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          <StatCard label="Total Explorers"   value={leaderboard.total_users > 0 ? leaderboard.total_users.toLocaleString() : '—'}         icon={User}      color="#1a73e8" />
          <StatCard label="Total Discoveries" value={leaderboard.total_discoveries > 0 ? leaderboard.total_discoveries.toLocaleString() : '—'} icon={Trophy}    color="#f59e0b" />
          <StatCard label="Top Discoverer"    value={leaderboard.rankings[0]?.user_id ?? '—'}                                                   icon={Trophy}    color="#7c3aed" />
          <StatCard label="Period"            value={leaderboard.period}                                                                          icon={Telescope} color="#16a34a" />
        </div>

        {/* Main content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Rankings — 3 cols */}
          <div className="lg:col-span-3">
            {loading ? (
              <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-12 flex items-center justify-center">
                <Loader2 className="w-8 h-8 text-[#1a73e8] animate-spin" />
              </div>
            ) : (
              <LeaderboardTable data={leaderboard} currentUserId={currentUserId} />
            )}
          </div>

          {/* Sidebar — 1 col */}
          <div className="space-y-5">
            <RankLegend />

            {/* Points guide */}
            <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-5">
              <h3 className="font-medium text-[#202124] text-sm mb-4">Points Guide</h3>
              <div className="space-y-2">
                {[
                  { action: 'Submit candidate',         pts: '+10' },
                  { action: 'Candidate confirmed',      pts: '+50' },
                  { action: 'Discovery verified',       pts: '+100' },
                  { action: 'High-priority discovery',  pts: '+200' },
                  { action: 'First of its type',        pts: '+500' },
                  { action: 'Verify others (×3)',       pts: '+15' },
                ].map(r => (
                  <div key={r.action} className="flex items-center justify-between text-xs">
                    <span className="text-[#5f6368]">{r.action}</span>
                    <span className="font-medium text-[#16a34a]">{r.pts}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* CTA */}
            <div className="bg-[#202124] rounded-2xl p-5 text-white">
              <Telescope className="w-6 h-6 mb-3 opacity-80" />
              <p className="text-sm font-medium mb-1">Start discovering</p>
              <p className="text-xs opacity-70 mb-4">
                5 free discovery runs per month. No PhD or GPU required.
              </p>
              <Link href="/discover" className="block text-center bg-white text-[#202124] rounded-lg py-2 text-sm font-medium hover:bg-[#f1f3f4] transition-colors">
                Open Sky Map
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
